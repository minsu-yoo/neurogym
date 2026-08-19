"""RNN_tutorial_fixed.ipynb를 GPU(CUDA)에서도 돌아가게 수정한 사본 생성.

원본은 device 처리가 없어 항상 CPU에서 돈다. 아래 패치는 torch.cuda.is_available()
분기라 GPU가 없는 환경에서도 그대로 동작한다.

주의: 이 튜토리얼 크기(hidden=128, batch=16, seq=100)에서는 GPU가 오히려 느리다.
CTRNN이 시간 스텝 100번을 파이썬 for 루프로 돌기 때문에 스텝당 연산량(16x128 행렬곱)이
커널 실행 오버헤드보다 작다. RTX 2080 실측 기준 iteration당 CPU 35.8ms / GPU 53.1ms.
hidden_size를 키우면 역전된다 (hidden=1024, batch=256에서 CPU 1094ms / GPU 68ms).
"""

import json
from pathlib import Path

SRC = Path("RNN_tutorial_fixed.ipynb")
DST = Path("RNN_tutorial_gpu.ipynb")

# (셀 인덱스, 원본 조각, 대체 조각) — 정확히 1회 치환되어야 함
PATCHES = [
    # --- cell 27: 모델을 device로 올리고, 학습 배치도 device로 ---
    (27, "net = RNNNet(input_size=input_size, hidden_size=hidden_size,\n"
         "             output_size=output_size, dt=env.dt)",
         "# .to(device): 파라미터를 GPU 메모리로 옮긴다. 반드시 optimizer 생성 전에 해야 한다.\n"
         "net = RNNNet(input_size=input_size, hidden_size=hidden_size,\n"
         "             output_size=output_size, dt=env.dt).to(device)"),
    (27, "        inputs = torch.from_numpy(inputs).type(torch.float)\n"
         "        labels = torch.from_numpy(labels.flatten()).type(torch.long)",
         "        # 입력/정답도 모델과 같은 device에 있어야 한다\n"
         "        inputs = torch.from_numpy(inputs).type(torch.float).to(device)\n"
         "        labels = torch.from_numpy(labels.flatten()).type(torch.long).to(device)"),

    # --- cell 29: 분석 루프. GPU 텐서는 .cpu()를 거쳐야 numpy로 변환된다 ---
    (29, "    inputs = torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float)",
         "    inputs = torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float).to(device)"),
    (29, "    action_pred = action_pred.detach().numpy()[:, 0, :]",
         "    # GPU 텐서는 .numpy()를 바로 못 부른다. .cpu()로 먼저 내려야 한다.\n"
         "    action_pred = action_pred.detach().cpu().numpy()[:, 0, :]"),
    (29, "    rnn_activity = rnn_activity[:, 0, :].detach().numpy()",
         "    rnn_activity = rnn_activity[:, 0, :].detach().cpu().numpy()"),
]

nb = json.loads(SRC.read_text(encoding="utf-8"))
cells = nb["cells"]

for idx, old, new in PATCHES:
    src = "".join(cells[idx]["source"])
    if src.count(old) != 1:
        raise SystemExit(f"cell {idx}: 패턴이 {src.count(old)}회 발견됨 (1회여야 함)\n{old!r}")
    cells[idx]["source"] = (src.replace(old, new)).splitlines(keepends=True)

# 공통 import 셀(7)에 device 정의 추가 — 이후 셀들이 모두 이 변수를 참조한다
src7 = "".join(cells[7]["source"])
cells[7]["source"] = (
    src7.rstrip("\n")
    + "\n\n# GPU가 있으면 GPU, 없으면 CPU로 자동 선택\n"
      "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
      "print('device:', device)\n"
).splitlines(keepends=True)

# 기존 실행 결과 제거 (깨끗한 상태에서 실행하기 위해)
for c in cells:
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None

DST.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"생성: {DST}  (패치 {len(PATCHES)}건 + device 정의)")
