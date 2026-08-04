"""RNN_tutorial.ipynb를 neurogym 2.3.1 / gymnasium 0.29 API에 맞게 수정한 사본 생성."""

import json
from pathlib import Path

SRC = Path("RNN_tutorial.ipynb")
DST = Path("RNN_tutorial_fixed.ipynb")

# (셀 인덱스, 원본 조각, 대체 조각) — 정확히 1회 치환되어야 함
PATCHES = [
    # --- cell 19: 구 gym 제거, wrapper 벗기기 ---
    (19, "import gym  # package for RL environments\nimport neurogym as ngym",
         "import neurogym as ngym\nfrom neurogym.core import TrialEnv"),
    (19, "# Boilerplate gym\nenv = gym.make(task_name, **kwargs)",
         "# neurogym 2.x: gym.make -> ngym.make\n"
         "# .unwrapped: OrderEnforcing 래퍼를 벗겨야 env.ob/env.gt/env.t 접근 시 경고가 안 뜬다\n"
         "env = ngym.make(task_name, **kwargs).unwrapped"),

    # --- cell 21: TrialEnv 위치 변경 + step 반환값 5개 ---
    (21, "class PerceptualDecisionMaking(ngym.TrialEnv):",
         "class PerceptualDecisionMaking(TrialEnv):"),
    (21, "        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}",
         "        # gymnasium: done -> (terminated, truncated) 로 분리되어 반환값이 5개\n"
         "        return self.ob_now, reward, False, False, {'new_trial': new_trial, 'gt': gt}"),

    # --- cell 29: no_step이 options 딕셔너리로 이동 ---
    (29, "env.reset(no_step=True)",
         "env.reset(options={'no_step': True})"),

    # --- cell 50: 두 번째 커스텀 환경도 동일 수정 ---
    (50, "class SimpleVisualDecisionMaking(ngym.TrialEnv):",
         "class SimpleVisualDecisionMaking(TrialEnv):"),
    (50, "        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}",
         "        return self.ob_now, reward, False, False, {'new_trial': new_trial, 'gt': gt}"),

    # --- cell 52: reset options + step 반환값 5개 ---
    (52, "vis_env.reset(no_step=True)",
         "vis_env.reset(options={'no_step': True})"),
    (52, "    ob, _, _, _ = vis_env.step(action=0)  # keep choosing action 0",
         "    ob, _, _, _, _ = vis_env.step(action=0)  # keep choosing action 0"),

    # --- cell 54 ---
    (54, "env = gym.make('DelayComparison-v0')",
         "env = ngym.make('DelayComparison-v0').unwrapped"),
]

nb = json.loads(SRC.read_text())
cells = nb["cells"]

for idx, old, new in PATCHES:
    src = "".join(cells[idx]["source"])
    if src.count(old) != 1:
        raise SystemExit(f"cell {idx}: 패턴이 {src.count(old)}회 발견됨 (1회여야 함)\n{old!r}")
    cells[idx]["source"] = (src.replace(old, new)).splitlines(keepends=True)

# 폰트 경고 억제를 공통 import 셀(7)에 추가
src7 = "".join(cells[7]["source"])
cells[7]["source"] = (
    "%matplotlib inline\n\n"
    + src7.rstrip("\n")
    + "\n\n# macOS에는 Noto Sans가 없어 findfont 경고가 뜬다. 렌더링은 DejaVu Sans로 정상 처리됨.\n"
      "import logging\n"
      'logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)\n'
).splitlines(keepends=True)

# 기존 실행 결과 제거 (깨끗한 상태에서 실행하기 위해)
for c in cells:
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None

DST.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"생성: {DST}  (패치 {len(PATCHES)}건 + 폰트 경고 억제)")
