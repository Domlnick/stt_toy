#!/usr/bin/env python3
"""STT 결과(txt)를 Markdown 회의록 템플릿으로 변환한다.

- output/txt/*.txt 를 읽어 output/md/<이름>.md 를 생성한다.
- 표준 라이브러리만 사용한다 (CLAUDES.md 10.3: 불필요한 의존성 금지).
- 자동 요약/화자분리는 범위 외. 빈 템플릿 섹션만 제공한다 (CLAUDES.md 9.2).

사용법:
    python3 src/postprocess.py                # output/txt 전체 변환
    python3 src/postprocess.py <파일.txt> ...  # 지정 파일만 변환
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 = 이 파일의 부모의 부모 (src/postprocess.py 기준)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TXT_DIR = PROJECT_ROOT / "output" / "txt"
MD_DIR = PROJECT_ROOT / "output" / "md"

# config.sh 와 일치시키는 기본값 (메타데이터 표기용)
MODEL_NAME = "small"
LANG_OPT = "ko"


def read_text(path: Path) -> str:
    """txt 파일 내용을 읽는다. 인코딩 문제는 호출부에서 처리."""
    return path.read_text(encoding="utf-8").strip()


def build_markdown(name: str, body: str) -> str:
    """원문 + 빈 회의록 템플릿 섹션을 가진 Markdown 문자열을 만든다."""
    created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# {name}",
        "",
        "> 메타데이터",
        f"> - 생성시간: {created}",
        f"> - 모델: {MODEL_NAME}",
        f"> - 언어: {LANG_OPT}",
        "",
        "## 요약",
        "",
        "(작성 예정)",
        "",
        "## 안건",
        "",
        "- ",
        "",
        "## 이슈",
        "",
        "- ",
        "",
        "## 결정사항",
        "",
        "- ",
        "",
        "## TODO",
        "",
        "- [ ] ",
        "",
        "## 전체 원문",
        "",
        body if body else "(원문 없음)",
        "",
    ]
    return "\n".join(lines)


def convert_one(txt_path: Path) -> Path:
    """txt 파일 1개를 md 로 변환하고 출력 경로를 반환한다."""
    name = txt_path.stem
    body = read_text(txt_path)
    md = build_markdown(name, body)

    MD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MD_DIR / f"{name}.md"
    out_path.write_text(md, encoding="utf-8")
    return out_path


def collect_targets(argv: list[str]) -> list[Path]:
    """인자가 있으면 그 파일들, 없으면 output/txt 전체를 대상으로 한다."""
    if argv:
        return [Path(p) for p in argv]
    if not TXT_DIR.is_dir():
        return []
    return sorted(TXT_DIR.glob("*.txt"))


def main() -> int:
    targets = collect_targets(sys.argv[1:])
    if not targets:
        print(f"[postprocess] 변환할 txt 파일이 없다: {TXT_DIR}")
        return 0

    ok = 0
    fail = 0
    for txt_path in targets:
        if not txt_path.is_file():
            print(f"[postprocess] 파일 없음, 건너뜀: {txt_path}", file=sys.stderr)
            fail += 1
            continue
        try:
            out_path = convert_one(txt_path)
            print(f"[postprocess] 변환 완료: {out_path}")
            ok += 1
        except Exception as exc:  # 예외 처리 (CLAUDES.md 10.1)
            print(f"[postprocess] 변환 실패: {txt_path} ({exc})", file=sys.stderr)
            fail += 1

    print(f"[postprocess] 요약 | 성공={ok} 실패={fail}")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
