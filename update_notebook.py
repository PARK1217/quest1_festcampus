import nbformat

def update_notebook(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # 마지막 두 셀(마크다운 섹션 8과 해당 코드 셀)을 찾아서 교체하거나 새로 작성
    # 섹션 8 제목 셀 찾기
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'markdown' and '## 8. 비즈니스 시나리오별 임계값' in cell.source:
            # 해당 셀부터 끝까지 삭제 후 새로 추가
            nb.cells = nb.cells[:i]
            break

    # 섹션 8 마크다운 셀 추가
    nb.cells.append(nbformat.v4.new_markdown_cell("## 8. 비즈니스 시나리오별 임계값(Threshold) 조정"))

    # 섹션 8 코드 셀 추가 (50% vs 70% 비교)
    code_source = """# --- [비즈니스 시나리오별 임계값 비교: 50% vs 70%] ---
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. 임계값 70% 조정 결과 계산
threshold = 0.7
y_proba_best = best_model.predict_proba(X_test_scaled)[:, 1]
y_pred_70 = (y_proba_best >= threshold).astype(int)

print(f'=== {best_name} 성능 비교 ===')
print(f'[기본 50%] F1: {f1_score(y_test, y_pred_best):.4f}, Precision: {precision_score(y_test, y_pred_best):.4f}, Recall: {recall_score(y_test, y_pred_best):.4f}')
print(f'[보수 70%] F1: {f1_score(y_test, y_pred_70):.4f}, Precision: {precision_score(y_test, y_pred_70):.4f}, Recall: {recall_score(y_test, y_pred_70):.4f}')

# 2. 혼동 행렬 양옆 비교 시각화
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 왼쪽: 임계값 50% (기본)
cm_50 = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm_50, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['거절', '승인'], yticklabels=['거절', '승인'])
axes[0].set_title('임계값 50% (기본 승인 전략)')
axes[0].set_xlabel('예측')
axes[0].set_ylabel('실제')

# 오른쪽: 임계값 70% (보수적 관리)
cm_70 = confusion_matrix(y_test, y_pred_70)
sns.heatmap(cm_70, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
            xticklabels=['거절', '승인'], yticklabels=['거절', '승인'])
axes[1].set_title('임계값 70% (보수적 리스크 관리)')
axes[1].set_xlabel('예측')
axes[1].set_ylabel('실제')

plt.tight_layout()
plt.show()"""
    nb.cells.append(nbformat.v4.new_code_cell(code_source))

    with open(file_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"Successfully updated {file_path}")

if __name__ == "__main__":
    update_notebook('credit_scoring_model.ipynb')
