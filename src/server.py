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

# job_id -> {state, percent, text, error}
#   state: queued | loading | transcribing | diarizing | done | error
JOBS = {}
JOBS_LOCK = threading.Lock()


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
  body { font-family: -apple-system, sans-serif; max-width: 760px; margin: 40px auto; padding: 0 16px; color: #222; }
  h1 { font-size: 20px; }
  #drop { border: 2px dashed #bbb; border-radius: 10px; padding: 36px; text-align: center; color: #666; cursor: pointer; transition: .15s; }
  #drop.hover { border-color: #4a90d9; background: #f0f7ff; color: #4a90d9; }
  #file { display: none; }
  .opt { display: block; margin: 12px 2px 0; font-size: 13px; color: #555; cursor: pointer; }
  #status { margin: 14px 0 6px; color: #4a90d9; min-height: 20px; font-size: 14px; }
  progress { width: 100%; height: 16px; display: none; }
  #out { white-space: pre-wrap; background: #f7f7f8; border: 1px solid #e3e3e6; border-radius: 8px; padding: 16px; min-height: 80px; margin-top: 14px; }
  .meta { color: #888; font-size: 13px; margin-top: 10px; }
</style>
</head>
<body>
  <h1>로컬 STT — 음성 → 텍스트</h1>
  <div id="drop">여기로 .m4a 파일을 끌어다 놓거나 클릭해서 선택</div>
  <input type="file" id="file" accept=".m4a,audio/*">
  <label class="opt"><input type="checkbox" id="diar"> 화자 구분 (느림, 화자 바뀔 때 줄바꿈·동시발화 표시)</label>
  <div id="status"></div>
  <progress id="bar" max="100" value="0"></progress>
  <div class="meta" id="meta"></div>
  <div id="out">결과가 여기 표시됩니다.</div>

<script>
const drop = document.getElementById('drop');
const file = document.getElementById('file');
const statusEl = document.getElementById('status');
const bar = document.getElementById('bar');
const out = document.getElementById('out');
const meta = document.getElementById('meta');

const diar = document.getElementById('diar');

const LABEL = {
  queued: '대기 중…',
  loading: '모델 로딩 중…',
  transcribing: '음성 인식 중…',
  diarizing: '화자 구분 중…',
  done: '완료',
  error: '실패',
};

drop.addEventListener('click', () => file.click());
drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('hover'); });
drop.addEventListener('dragleave', () => drop.classList.remove('hover'));
drop.addEventListener('drop', e => {
  e.preventDefault(); drop.classList.remove('hover');
  if (e.dataTransfer.files.length) upload(e.dataTransfer.files[0]);
});
file.addEventListener('change', () => { if (file.files.length) upload(file.files[0]); });

async function upload(f) {
  meta.textContent = '';
  out.textContent = '';
  bar.style.display = 'block';
  bar.removeAttribute('value');        // 시작은 indeterminate(움직이는 바)
  statusEl.textContent = '업로드 중…';
  const fd = new FormData();
  fd.append('audio', f, f.name);
  fd.append('diarize', diar.checked ? '1' : '0');
  let job;
  try {
    const res = await fetch('/transcribe', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok || data.error) { fail(data.error || ('오류: ' + res.status)); return; }
    job = data.job_id;
  } catch (err) { fail('업로드 실패: ' + err); return; }
  poll(job, f.name);
}

function fail(msg) {
  bar.style.display = 'none';
  statusEl.textContent = '실패';
  out.textContent = msg;
}

function poll(job, fname) {
  const timer = setInterval(async () => {
    let d;
    try {
      const res = await fetch('/progress?id=' + encodeURIComponent(job));
      d = await res.json();
    } catch (err) { return; }   // 일시 오류는 다음 폴링에서 재시도

    statusEl.textContent = LABEL[d.state] || d.state;
    if (typeof d.percent === 'number' && d.state === 'transcribing') {
      bar.value = d.percent;            // 실제 % 표시
      statusEl.textContent = LABEL.transcribing + ' ' + d.percent + '%';
    } else if (d.state === 'queued' || d.state === 'loading' || d.state === 'diarizing') {
      bar.removeAttribute('value');     // indeterminate (화자 구분은 진행률 불확정)
    }

    if (d.state === 'done') {
      clearInterval(timer);
      bar.value = 100;
      setTimeout(() => { bar.style.display = 'none'; }, 400);
      statusEl.textContent = '완료';
      meta.textContent = '파일: ' + fname;
      out.textContent = d.text || '(빈 결과)';
    } else if (d.state === 'error') {
      clearInterval(timer);
      fail(d.error || '처리 실패');
    }
  }, 500);
}
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

    duration = 0.0
    last_lines = []
    for line in proc.stdout:
        last_lines.append(line.rstrip())
        if len(last_lines) > 40:
            last_lines.pop(0)

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
    set_job(job_id, state="done", percent=100, text=text)


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
