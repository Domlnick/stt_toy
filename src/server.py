#!/usr/bin/env python3
"""로컬 STT 웹 UI (진행률 표시).

브라우저에서 음성 파일을 업로드하면 기존 처리 스크립트
(scripts/transcribe_one.sh)를 백그라운드로 실행하고,
whisper 의 진행률(%)을 폴링으로 받아 progress bar 로 보여준다.

- Python 표준 라이브러리만 사용한다 (CLAUDES.md 10.3).
- 127.0.0.1 에만 바인딩한다. 외부 업로드/접속 없음 (CLAUDES.md 10.5).
- 업로드 파일은 input/ 에 저장되고 원본은 삭제되지 않는다.

동작 구조:
    POST /transcribe  -> 파일 저장 후 작업(job) 시작, job_id 반환
    GET  /progress?id -> 해당 작업의 상태/진행률/결과 폴링

사용법:
    python3 src/server.py            # http://127.0.0.1:8000
    python3 src/server.py 9000       # 포트 지정
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse, parse_qs

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "input"
TXT_DIR = PROJECT_ROOT / "output" / "txt"
MLX_SCRIPT = PROJECT_ROOT / "src" / "mlx_transcribe.py"

HOST = "127.0.0.1"
DEFAULT_PORT = 8000

# mlx_transcribe.py 출력 파싱
#   "DURATION 118.7"
#   "[00:30.000 --> 00:35.500]  text"  -> 종료 타임스탬프로 진행률 계산
DUR_RE = re.compile(r"^DURATION\s+([\d.]+)")
SEG_RE = re.compile(r"-->\s*(\d+):(\d+(?:\.\d+)?)\]")
PHASE_RE = re.compile(r"^PHASE\s+(\w+)")
DIAR_ERR_RE = re.compile(r"^DIARIZE_ERROR\s+(.*)")

# job_id -> {state, percent, text, error}
#   state: queued | loading | transcribing | diarizing | done | error | cancelled
JOBS = {}
JOBS_LOCK = threading.Lock()

# job_id -> Popen (취소 시 종료용). JSON 직렬화 대상(JOBS)과 분리.
PROCS = {}
PROCS_LOCK = threading.Lock()


def set_proc(job_id: str, proc):
    with PROCS_LOCK:
        PROCS[job_id] = proc


def pop_proc(job_id: str):
    with PROCS_LOCK:
        return PROCS.pop(job_id, None)


def cancel_job(job_id: str) -> bool:
    """실행 중인 작업의 서브프로세스를 종료한다. 취소 처리되면 True."""
    with PROCS_LOCK:
        proc = PROCS.get(job_id)
    if proc is None:
        return False
    set_job(job_id, state="cancelled")   # worker 가 error 로 덮어쓰지 않도록 먼저 표시
    try:
        proc.terminate()
    except Exception:
        pass
    return True


def set_job(job_id: str, **fields):
    with JOBS_LOCK:
        JOBS.setdefault(job_id, {}).update(fields)


def get_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        return dict(job) if job else None


INDEX_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>로컬 STT</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; color: #222;
         background: #fff; height: 100vh; display: flex; flex-direction: column; }
  header { text-align: center; padding: 16px; border-bottom: 1px solid #eee; }
  header h1 { margin: 0; font-size: 20px; }
  main { flex: 1; display: flex; gap: 16px; padding: 16px; min-height: 0; }
  .pane { flex: 1; display: flex; flex-direction: column; border: 1px solid #e3e3e6;
          border-radius: 12px; padding: 16px; min-height: 0; }
  .paneTitle { font-size: 14px; font-weight: 600; color: #444; margin: 0 2px 12px; }

  /* 왼쪽: 업로드 */
  .drop { flex: 1; border: 2px dashed #cfd4da; border-radius: 12px; display: flex;
          align-items: center; justify-content: center; text-align: center; color: #8a9099;
          cursor: pointer; padding: 16px; transition: .15s; min-height: 0; line-height: 1.6;
          word-break: break-all; }
  .drop.hover { border-color: #1aa179; background: #f0faf6; color: #1aa179; }
  .drop.has { color: #222; border-style: solid; border-color: #1aa179; background: #f7fbf9; }
  .dur { margin-top: 10px; font-size: 13px; color: #555; min-height: 18px; text-align: center; }
  .opt { display: block; margin: 8px 0 12px; font-size: 13px; color: #555; cursor: pointer; }
  .opt span { color: #aab; }
  .btnRow { display: flex; gap: 10px; height: 48px; }
  .btn { flex: 1; border: none; border-radius: 8px; font-size: 15px; cursor: pointer; color: #fff; }
  .btn.green { background: #1aa179; }
  .btn.gray { background: #9aa0a6; }
  .btn:disabled { opacity: .45; cursor: not-allowed; }
  .left.disabled { pointer-events: none; opacity: .55; }

  /* 오른쪽: 결과 */
  .out { flex: 1; white-space: pre-wrap; font-size: 11px; line-height: 1.6; background: #f7f7f8;
         border: 1px solid #e3e3e6; border-radius: 8px; padding: 12px; overflow-y: auto;
         min-height: 0; color: #333; }
  .resFooter { margin-top: 10px; display: flex; align-items: center; justify-content: space-between;
               gap: 10px; flex-wrap: wrap; }
  .stats { font-size: 12px; color: #777; }
  .resActions { display: flex; gap: 8px; }
  .icon { font-size: 13px; padding: 6px 10px; border-radius: 8px; border: 1px solid #d0d4da;
          background: #fff; color: #333; cursor: pointer; }
  .btn.small { font-size: 13px; padding: 6px 12px; border-radius: 8px; background: #1aa179;
               color: #fff; border: none; cursor: pointer; }
  .icon:disabled, .btn.small:disabled { opacity: .45; cursor: not-allowed; }

  /* 작업 취소 버튼: 항상 화면 우상단 고정 */
  #cancelBtn { position: fixed; top: 14px; right: 16px; z-index: 1100; background: #e5484d;
               color: #fff; border: none; border-radius: 8px; padding: 8px 14px; font-size: 14px;
               cursor: pointer; display: none; box-shadow: 0 2px 8px rgba(0,0,0,.25); }

  /* 진행 오버레이: 전체 어둡게 + 중앙 progress */
  .overlay { position: fixed; inset: 0; background: rgba(0,0,0,.62); z-index: 1000;
             display: none; align-items: center; justify-content: center; }
  .overlay.show { display: flex; }
  .ov-box { background: #fff; border-radius: 14px; padding: 26px 30px; width: min(440px, 82vw);
            text-align: center; box-shadow: 0 8px 30px rgba(0,0,0,.3); }
  .status { font-size: 15px; color: #222; margin-bottom: 14px; min-height: 20px; }
  .ov-box progress { width: 100%; height: 16px; }
</style>
</head>
<body>
  <button id="cancelBtn">작업 취소하기</button>
  <header><h1>로컬 STT</h1></header>

  <main>
    <section class="pane left" id="leftPane">
      <div class="paneTitle">음성 파일 첨부</div>
      <div id="drop" class="drop">여기로 .m4a 파일을<br>끌어다 놓거나 클릭해서 선택</div>
      <input type="file" id="file" accept=".m4a,audio/*" hidden>
      <div id="dur" class="dur"></div>
      <label class="opt"><input type="checkbox" id="diar"> 화자 구분 <span>(시간이 오래 걸립니다)</span></label>
      <div class="btnRow">
        <button id="extractBtn" class="btn green" disabled>추출하기</button>
        <button id="clearBtn" class="btn gray" disabled>삭제하기</button>
      </div>
    </section>

    <section class="pane right">
      <div class="paneTitle">추출 결과</div>
      <div id="out" class="out">결과가 여기 표시됩니다.</div>
      <div class="resFooter">
        <div class="stats" id="stats"></div>
        <div class="resActions">
          <button id="copyBtn" class="icon" disabled>📋 복사</button>
          <button id="dlBtn" class="btn small" disabled>TXT 다운로드</button>
        </div>
      </div>
    </section>
  </main>

  <div id="overlay" class="overlay">
    <div class="ov-box">
      <div id="status" class="status"></div>
      <progress id="bar" max="100" value="0"></progress>
    </div>
  </div>

<script>
const $ = id => document.getElementById(id);
const drop = $('drop'), file = $('file'), durEl = $('dur'), diar = $('diar');
const extractBtn = $('extractBtn'), clearBtn = $('clearBtn'), leftPane = $('leftPane');
const out = $('out'), stats = $('stats'), copyBtn = $('copyBtn'), dlBtn = $('dlBtn');
const overlay = $('overlay'), statusEl = $('status'), bar = $('bar'), cancelBtn = $('cancelBtn');

let selectedFile = null, curJob = null, timer = null, resultText = '', baseName = 'result';

const LABEL = {
  queued: '대기 중…', loading: '모델 로딩 중…', transcribing: '음성 → 텍스트 추출 중…',
  diarizing: '화자 구분 중…', done: '완료', error: '실패', cancelled: '취소됨',
};
const DROP_DEFAULT = '여기로 .m4a 파일을<br>끌어다 놓거나 클릭해서 선택';

function fmtDur(sec) {
  sec = Math.round(sec); const m = Math.floor(sec / 60), s = sec % 60;
  return String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0');
}
function escapeHtml(s) {
  return s.replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}

/* 파일 선택 / 드래그앤드롭 */
drop.addEventListener('click', () => { if (!leftPane.classList.contains('disabled')) file.click(); });
drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('hover'); });
drop.addEventListener('dragleave', () => drop.classList.remove('hover'));
drop.addEventListener('drop', e => {
  e.preventDefault(); drop.classList.remove('hover');
  if (e.dataTransfer.files.length) pick(e.dataTransfer.files[0]);
});
file.addEventListener('change', () => { if (file.files.length) pick(file.files[0]); });

function pick(f) {
  selectedFile = f;
  drop.classList.add('has');
  drop.innerHTML = '📄 ' + escapeHtml(f.name);
  extractBtn.disabled = false; clearBtn.disabled = false;
  durEl.textContent = '길이 측정 중…';
  // 업로드 전 브라우저에서 즉시 길이 측정 (HTML5, 파일 길이와 무관하게 빠름)
  const a = document.createElement('audio'); a.preload = 'metadata';
  a.onloadedmetadata = () => { durEl.textContent = '음성 길이 ' + fmtDur(a.duration); URL.revokeObjectURL(a.src); };
  a.onerror = () => { durEl.textContent = '길이 정보 없음'; };
  a.src = URL.createObjectURL(f);
}

clearBtn.addEventListener('click', resetInput);
function resetInput() {
  selectedFile = null; file.value = '';
  drop.classList.remove('has'); drop.innerHTML = DROP_DEFAULT;
  durEl.textContent = ''; extractBtn.disabled = true; clearBtn.disabled = true;
}

/* 추출 시작 */
extractBtn.addEventListener('click', startExtract);
async function startExtract() {
  if (!selectedFile) return;
  baseName = selectedFile.name.replace(/\.[^.]+$/, '') || 'result';
  setProcessing(true, '업로드 중…');
  const fd = new FormData();
  fd.append('audio', selectedFile, selectedFile.name);
  fd.append('diarize', diar.checked ? '1' : '0');
  let job;
  try {
    const res = await fetch('/transcribe', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok || data.error) { failOverlay(data.error || ('오류: ' + res.status)); return; }
    job = data.job_id;
  } catch (err) { failOverlay('업로드 실패: ' + err); return; }
  curJob = job; poll(job);
}

function setProcessing(on, label) {
  if (on) {
    leftPane.classList.add('disabled');          // 추출 중 업로드 영역 비활성화
    overlay.classList.add('show');
    cancelBtn.style.display = 'block';
    bar.removeAttribute('value'); statusEl.textContent = label || '';
    copyBtn.disabled = true; dlBtn.disabled = true;
  } else {
    leftPane.classList.remove('disabled');
    overlay.classList.remove('show');
    cancelBtn.style.display = 'none';
  }
}

function poll(job) {
  timer = setInterval(async () => {
    let d;
    try { const res = await fetch('/progress?id=' + encodeURIComponent(job)); d = await res.json(); }
    catch (err) { return; }   // 일시 오류는 다음 폴링에서 재시도
    statusEl.textContent = LABEL[d.state] || d.state;
    if (typeof d.percent === 'number' && d.state === 'transcribing') {
      bar.value = d.percent;
      statusEl.textContent = '음성 → 텍스트 추출 중… ' + d.percent + '%';
    } else if (d.state === 'queued' || d.state === 'loading' || d.state === 'diarizing') {
      bar.removeAttribute('value');   // indeterminate
    }
    if (d.state === 'done') { stop(); showResult(d.text || '', d.warning); }
    else if (d.state === 'error') { stop(); failOverlay(d.error || '처리 실패'); }
    else if (d.state === 'cancelled') { stop(); setProcessing(false); }
  }, 500);
}
function stop() { if (timer) { clearInterval(timer); timer = null; } curJob = null; }

function showResult(text, warning) {
  bar.value = 100; resultText = text;
  setTimeout(() => setProcessing(false), 300);
  out.textContent = text || '(빈 결과)';
  const words = (text.trim().match(/\S+/g) || []).length;
  const chars = text.replace(/\s/g, '').length;
  let s = '총 단어 수 : ' + words + ' 개  /  총 문자 수 : ' + chars + ' 개';
  if (warning) s = '⚠️ ' + warning + '\\n' + s;
  stats.textContent = s;
  stats.style.color = warning ? '#e5484d' : '#777';
  stats.style.whiteSpace = 'pre-line';
  copyBtn.disabled = !text; dlBtn.disabled = !text;
}

function failOverlay(msg) {
  stop(); setProcessing(false);
  out.textContent = msg; stats.textContent = ''; resultText = '';
  copyBtn.disabled = true; dlBtn.disabled = true;
}

/* 작업 취소 */
cancelBtn.addEventListener('click', async () => {
  const id = curJob;
  if (!id) { setProcessing(false); return; }
  try { await fetch('/cancel?id=' + encodeURIComponent(id)); } catch (e) {}
  stop(); setProcessing(false);
});

/* 결과 복사 */
copyBtn.addEventListener('click', async () => {
  if (!resultText) return;
  const flash = () => { copyBtn.textContent = '✅ 복사됨'; setTimeout(() => copyBtn.textContent = '📋 복사', 1500); };
  try { await navigator.clipboard.writeText(resultText); flash(); }
  catch (e) {
    const ta = document.createElement('textarea'); ta.value = resultText;
    document.body.appendChild(ta); ta.select(); document.execCommand('copy'); ta.remove(); flash();
  }
});

/* TXT 다운로드 */
dlBtn.addEventListener('click', () => {
  if (!resultText) return;
  const blob = new Blob([resultText], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob); const a = document.createElement('a');
  a.href = url; a.download = baseName + '.txt';
  document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
});
</script>
</body>
</html>
"""


def _disposition(header_blob: str):
    """Content-Disposition 줄에서 (name, filename) 추출. 없으면 (None, None)."""
    name = filename = None
    for line in header_blob.split("\r\n"):
        if not line.lower().startswith("content-disposition:"):
            continue
        for token in line.split(";"):
            token = token.strip()
            if token.startswith("name="):
                name = token[len("name="):].strip().strip('"')
            elif token.startswith("filename="):
                filename = unquote(token[len("filename="):].strip().strip('"'))
        break
    return name, filename


def parse_multipart(body: bytes, content_type: str):
    """multipart/form-data 본문에서 첫 파일과 텍스트 필드들을 추출한다.

    반환: (filename, file_bytes, fields)  — 파일 없으면 filename/file_bytes 는 None.
    fields = {필드명: 값(str)}  (파일이 아닌 일반 필드).
    """
    marker = "boundary="
    idx = content_type.find(marker)
    if idx == -1:
        return None, None, {}
    boundary = content_type[idx + len(marker):].strip().strip('"')
    delim = b"--" + boundary.encode()

    filename = None
    file_bytes = None
    fields = {}
    for part in body.split(delim):
        sep = part.find(b"\r\n\r\n")
        if sep == -1:
            continue
        header_blob = part[:sep].decode("utf-8", "replace")
        if "content-disposition:" not in header_blob.lower():
            continue
        data = part[sep + 4:]
        if data.endswith(b"\r\n"):
            data = data[:-2]

        name, fname = _disposition(header_blob)
        if fname is not None:               # 파일 필드(첫 번째만 채택)
            if file_bytes is None:
                filename, file_bytes = fname, data
        elif name:                          # 일반 텍스트 필드
            fields[name] = data.decode("utf-8", "replace").strip()
    return filename, file_bytes, fields


def safe_name(filename: str) -> str:
    """경로 조작 방지: 디렉터리 성분 제거."""
    name = Path(filename).name
    return name if name not in ("", ".", "..") else "upload.m4a"


def worker(job_id: str, audio_path: Path, diarize: bool = False):
    """백그라운드에서 MLX 트랜스크라이버를 실행하며 진행률을 갱신한다."""
    set_job(job_id, state="loading", percent=0)
    env = os.environ.copy()
    if diarize:
        env["STT_DIARIZE"] = "1"   # 화자 구분 활성화(토큰은 서버 환경의 HF_TOKEN 사용)
    try:
        proc = subprocess.Popen(
            # 같은 파이썬으로 실행(mlx_whisper 설치 보장), -u 로 출력 즉시 전달
            [sys.executable, "-u", str(MLX_SCRIPT), str(audio_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # 진행률/에러를 한 스트림으로
            text=True,
            bufsize=1,
            env=env,
        )
    except Exception as exc:
        set_job(job_id, state="error", error=f"스크립트 실행 실패: {exc}")
        return

    set_proc(job_id, proc)
    duration = 0.0
    warning = None
    last_lines = []
    for line in proc.stdout:
        last_lines.append(line.rstrip())
        if len(last_lines) > 40:
            last_lines.pop(0)

        em = DIAR_ERR_RE.match(line)
        if em:
            warning = "화자 구분을 적용하지 못했습니다(일반 텍스트로 저장): " + em.group(1).strip()
            continue

        dm = DUR_RE.match(line)
        if dm:
            duration = float(dm.group(1))
            set_job(job_id, state="transcribing", percent=0)
            continue

        pm = PHASE_RE.match(line)
        if pm:
            phase = pm.group(1)
            if phase == "diarize":
                set_job(job_id, state="diarizing")   # 화자 구분(진행률 불확정)
            else:
                set_job(job_id, state="transcribing")
            continue

        sm = SEG_RE.search(line)
        if sm and duration > 0:
            end_sec = int(sm.group(1)) * 60 + float(sm.group(2))
            pct = int(min(99, max(0, end_sec / duration * 100)))
            set_job(job_id, state="transcribing", percent=pct)
    proc.wait()
    pop_proc(job_id)

    # 사용자가 취소한 경우: 종료코드와 무관하게 취소 상태 유지.
    cur = get_job(job_id)
    if cur and cur.get("state") == "cancelled":
        return

    if proc.returncode != 0:
        # 스크립트가 출력한 마지막 줄들에서 원인 추정
        msg = "\n".join(last_lines[-6:]).strip() or f"종료코드 {proc.returncode}"
        set_job(job_id, state="error", error=msg)
        return

    txt_path = TXT_DIR / f"{audio_path.stem}.txt"
    if not txt_path.is_file():
        set_job(job_id, state="error", error=f"결과 파일을 찾을 수 없다: {txt_path}")
        return
    try:
        text = txt_path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        set_job(job_id, state="error", error=f"결과 읽기 실패: {exc}")
        return
    set_job(job_id, state="done", percent=100, text=text, warning=warning)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("[server] %s\n" % (fmt % args))

    def _send_json(self, code: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif parsed.path == "/progress":
            qs = parse_qs(parsed.query)
            job_id = (qs.get("id") or [""])[0]
            job = get_job(job_id)
            if not job:
                self._send_json(404, {"error": "unknown job"})
                return
            self._send_json(200, job)
        elif parsed.path == "/favicon.ico":
            self.send_response(204)   # 내용 없음(브라우저 favicon 요청 조용히 처리)
            self.end_headers()
        elif parsed.path == "/cancel":
            qs = parse_qs(parsed.query)
            job_id = (qs.get("id") or [""])[0]
            ok = cancel_job(job_id)
            self._send_json(200, {"cancelled": ok})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/transcribe":
            self._send_json(404, {"error": "not found"})
            return

        ctype = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in ctype:
            self._send_json(400, {"error": "multipart/form-data 가 아니다"})
            return

        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            self._send_json(400, {"error": "빈 요청"})
            return

        body = self.rfile.read(length)
        filename, data, fields = parse_multipart(body, ctype)
        if not filename or data is None:
            self._send_json(400, {"error": "업로드 파일을 찾을 수 없다"})
            return
        diarize = fields.get("diarize", "") in ("1", "true", "on", "yes")

        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        save_path = INPUT_DIR / safe_name(filename)
        try:
            save_path.write_bytes(data)
        except Exception as exc:
            self._send_json(500, {"error": f"파일 저장 실패: {exc}"})
            return

        job_id = uuid.uuid4().hex
        set_job(job_id, state="queued", percent=0)
        threading.Thread(
            target=worker, args=(job_id, save_path, diarize), daemon=True,
        ).start()
        self._send_json(202, {"job_id": job_id})


def main() -> int:
    port = DEFAULT_PORT
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"[server] 포트 번호가 잘못됨: {sys.argv[1]}", file=sys.stderr)
            return 1

    if not MLX_SCRIPT.is_file():
        print(f"[server] 스크립트가 없다: {MLX_SCRIPT}", file=sys.stderr)
        return 1

    # 화자 구분용 HF 토큰: 환경변수 우선, 없으면 프로젝트 루트 .hf_token 파일에서 읽는다.
    # (인라인 환경변수가 누락되는 일이 잦아 파일 방식을 폴백으로 둔다.)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or ""
    if not token:
        tok_file = PROJECT_ROOT / ".hf_token"
        if tok_file.is_file():
            token = tok_file.read_text(encoding="utf-8").strip()
            if token:
                os.environ["HF_TOKEN"] = token   # 자식(mlx_transcribe)에게 전달됨
    if token:
        masked = ("…" + token[-4:]) if len(token) >= 4 else "****"
        print(f"[server] 화자 구분 가능: HF_TOKEN 감지됨 ({masked})")
    else:
        print("[server] 화자 구분 비활성: HF_TOKEN 없음 "
              "(환경변수로 주거나 프로젝트 루트에 .hf_token 파일 생성)")

    server = ThreadingHTTPServer((HOST, port), Handler)
    print(f"[server] http://{HOST}:{port} 에서 실행 중 (Ctrl+C 종료)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] 종료")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
