import os
import time
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore')

import logging

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler("asym_fuzzy_training_log.txt", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# ==========================================
# 1. BỘ NẠP DỮ LIỆU
# ==========================================
class TCGALungDataset(Dataset):
    def __init__(self, csv_file, pt_dir):
        self.df = pd.read_csv(csv_file)
        self.pt_dir = pt_dir
        self.label_dict = {"LUAD": 1, "LUSC": 0}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wsi_id = row['wsi_id']
        label = self.label_dict[row['label']]
        pt_path = os.path.join(self.pt_dir, f"{wsi_id}.pt")
        features = torch.load(pt_path)
        return wsi_id, features, label

# ==========================================
# 2. MÔ HÌNH ASYMMETRIC FUZZY LAGRANGIAN MIL
# ==========================================
class AsymmetricFuzzyLagrangianMIL:
    def __init__(self, C=10.0, max_iter=5, lr_lambda=0.01):
        self.C = C 
        self.max_iter = max_iter
        self.lr_lambda = lr_lambda
        
        self.w = None
        self.b = None
        self.history = {'train_acc': [], 'test_acc': []}

    def _compute_fuzzy_weights(self, bags, bag_labels):
        """
        TỐI ƯU MỜ BẤT ĐỐI XỨNG (ASYMMETRIC FUZZY)
        - Túi LUSC (Âm tính): Trọng số 1.0 (Tin tưởng tuyệt đối)
        - Túi LUAD (Dương tính): Áp dụng Fuzzy để lọc nhiễu
        """
        bag_ids = list(bags.keys())
        X_list, bag_map, fuzzy_scores = [], [], []
        
        for i, bid in enumerate(bag_ids):
            insts = bags[bid]
            label = bag_labels[bid]
            
            if label == 0:
                # Túi âm tính: 100% các mảnh là âm tính, không cần làm mờ
                scores = np.ones(len(insts))
            else:
                # Túi dương tính: Chứa cả ung thư và nhiễu, cần làm mờ
                centroid = np.mean(insts, axis=0)
                distances = np.linalg.norm(insts - centroid, axis=1)
                max_d = np.max(distances) if np.max(distances) > 0 else 1.0
                
                # Cài đặt eps = 0.5
                eps = 0.5 
                scores = eps + (1 - eps) * (1 - (distances / max_d))
            
            for j, inst in enumerate(insts):
                X_list.append(inst)
                fuzzy_scores.append(scores[j])
                bag_map.append(i)
                
        return np.array(X_list), np.array(bag_map), np.array(fuzzy_scores)

    def fit(self, bags, bag_labels, eval_bags=None, eval_labels=None):
        bag_ids = list(bags.keys())
        y_bag = np.array([bag_labels[bid] for bid in bag_ids])
        pos_bag_indices = [i for i, y in enumerate(y_bag) if y == 1]
        neg_bag_indices = [i for i, y in enumerate(y_bag) if y == 0]

        # Khởi tạo X và S (Trọng số mờ bất đối xứng)
        self.X, self.bag_map, self.S = self._compute_fuzzy_weights(bags, bag_labels)
        self.num_instances = len(self.X)
        self.bag_indices_dict = {b_idx: np.where(self.bag_map == b_idx)[0] for b_idx in range(len(bag_ids))}
        
        self.lambdas = np.ones(len(pos_bag_indices))
        self.y = np.ones(self.num_instances)
        self.y[[i for i in range(self.num_instances) if self.bag_map[i] in neg_bag_indices]] = -1

        for l in range(self.max_iter):
            # 1. CỐ ĐỊNH y, GIẢI BÀI TOÁN TÌM SIÊU PHẲNG
            alpha_val = 1.0 / (self.C * self.num_instances) if self.C > 0 else 0.0001
            
            svc = SGDClassifier(
                loss='hinge', penalty='l2', alpha=alpha_val, 
                max_iter=100, tol=1e-3, n_jobs=-1, random_state=42, 
                class_weight='balanced', learning_rate='adaptive', eta0=0.01                 
            )
            # Truyền trọng số mờ S vào thuật toán SVM
            svc.fit(self.X, self.y, sample_weight=self.S)
            self.w, self.b = svc.coef_[0], svc.intercept_[0]

            # 2. CỐ ĐỊNH SIÊU PHẲNG, CẬP NHẬT LẠI NHÃN y
            scores = self.X.dot(self.w) + self.b
            new_y = self.y.copy()
            
            for idx_pos, b_idx in enumerate(pos_bag_indices):
                inst_ids = self.bag_indices_dict[b_idx]
                lam = self.lambdas[idx_pos]
                sub_scores = scores[inst_ids]
                fuzzy_s = self.S[inst_ids] # Lấy trọng số mờ
                
                # Hàm đối ngẫu Lagrange tích hợp trọng số mờ
                cost_pos = self.C * fuzzy_s * np.maximum(0, 1 - sub_scores) - lam
                cost_neg = self.C * fuzzy_s * np.maximum(0, 1 + sub_scores)
                new_y[inst_ids] = np.where(cost_pos < cost_neg, 1, -1)
            
            self.y = new_y

            # 3. CẬP NHẬT NHÂN TỬ LAGRANGE BẰNG SUB-GRADIENT
            for idx_pos, b_idx in enumerate(pos_bag_indices):
                inst_ids = self.bag_indices_dict[b_idx]
                num_pos = np.sum((self.y[inst_ids] + 1) / 2)
                self.lambdas[idx_pos] = max(0, self.lambdas[idx_pos] + self.lr_lambda * (1 - num_pos))

            # 4. THEO DÕI OVERFITTING BẰNG LEARNING CURVE
            train_scores = self.predict(bags)
            fpr, tpr, ths = roc_curve(y_bag, train_scores)
            best_th = ths[np.argmax(tpr - fpr)]
            
            train_pred = (train_scores >= best_th).astype(int)
            self.history['train_acc'].append(accuracy_score(y_bag, train_pred))

            if eval_bags is not None and eval_labels is not None:
                eval_scores = self.predict(eval_bags)
                eval_pred = (eval_scores >= best_th).astype(int)
                self.history['test_acc'].append(accuracy_score(eval_labels, eval_pred))

    def predict(self, bags):
        scores = []
        for bid, insts in bags.items():
            bag_score = np.max(insts.dot(self.w) + self.b)
            scores.append(bag_score)
        return np.array(scores)

# ==========================================
# 3. KỊCH BẢN THỰC THI (GRID SEARCH + 5-FOLD CV)
# ==========================================
if __name__ == "__main__":
    DUONG_DAN_CSV = "D:/LUNG/labels.csv"
    DUONG_DAN_THU_MOC_PT = "D:/LUNG/LUNG/pt_files"

    dataset = TCGALungDataset(csv_file=DUONG_DAN_CSV, pt_dir=DUONG_DAN_THU_MOC_PT)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    all_bags = {}
    all_labels = {}
    
    logging.info("Đang nạp dữ liệu Ảnh toàn lam (WSI)...")
    for i, (wsi_id, features, label) in enumerate(dataloader):
        bid = wsi_id[0]
        all_bags[bid] = features.squeeze(0).numpy()
        all_labels[bid] = int(label.item())
        if (i + 1) % 100 == 0: logging.info(f"Đã nạp {i+1} bệnh nhân...")

    bag_ids = np.array(list(all_bags.keys()))
    y_true = np.array([all_labels[b] for b in bag_ids])

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics = {'acc': [], 'f1': [], 'auc': []}
    
    all_true_labels = []
    all_pred_labels = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    all_learning_curves_train = []
    all_learning_curves_test = []
    
    fig_roc, ax_roc = plt.subplots(figsize=(10, 8)) 

    logging.info("\n" + "="*60)
    logging.info(" BẮT ĐẦU HUẤN LUYỆN ASYMMETRIC FUZZY MIL-RL ")
    logging.info("="*60)
    
    # Đã tăng dải C lên để bù đắp lực phạt cho hệ thống Fuzzy
    C_values_to_test = [2.0**i for i in range(-4, 11)] 
    logging.info(f"Lưới tham số C sẽ quét ({len(C_values_to_test)} giá trị): {C_values_to_test}")
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(bag_ids, y_true)):
        logging.info(f"\n*** FOLD {fold+1} ***")
        fold_start_time = time.time()
        
        train_bags = {bag_ids[i]: all_bags[bag_ids[i]] for i in train_idx}
        train_labels = {bag_ids[i]: all_labels[bag_ids[i]] for i in train_idx}
        test_bags = {bag_ids[i]: all_bags[bag_ids[i]] for i in test_idx}
        test_labels = [all_labels[bag_ids[i]] for i in test_idx]

        logging.info("   [1/4] Đang học quy luật chuẩn hóa và nén PCA (400 chiều)...")
        scaler = StandardScaler()
        for insts in train_bags.values():
            scaler.partial_fit(insts)
            
        pca = IncrementalPCA(n_components=400, batch_size=2048)
        for insts in train_bags.values():
            pca.partial_fit(scaler.transform(insts))
            
        scaled_pca_train_bags = {k: pca.transform(scaler.transform(v)) for k, v in train_bags.items()}
        scaled_pca_test_bags = {k: pca.transform(scaler.transform(v)) for k, v in test_bags.items()}

        logging.info("   [2/4] Đang quét Grid Search tìm C tối ưu...")
        train_bag_ids = list(train_bags.keys())
        train_labels_list = [train_labels[bid] for bid in train_bag_ids]
        
        sub_train_ids, sub_val_ids = train_test_split(
            train_bag_ids, test_size=0.2, stratify=train_labels_list, random_state=42
        )
        
        sub_train_bags = {bid: scaled_pca_train_bags[bid] for bid in sub_train_ids}
        sub_train_labels = {bid: train_labels[bid] for bid in sub_train_ids}
        sub_val_bags = {bid: scaled_pca_train_bags[bid] for bid in sub_val_ids}
        sub_val_labels = [train_labels[bid] for bid in sub_val_ids]

        best_C = C_values_to_test[0]
        best_val_auc = -1.0
        
        for c in C_values_to_test:
            val_model = AsymmetricFuzzyLagrangianMIL(C=c, max_iter=3, lr_lambda=0.01)
            val_model.fit(sub_train_bags, sub_train_labels)
            
            val_scores = val_model.predict(sub_val_bags)
            val_auc = roc_auc_score(sub_val_labels, val_scores)
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_C = c
                
        logging.info(f"   => ĐÃ CHỌN C TỐI ƯU: {best_C} (AUC trên tập Validation: {best_val_auc:.4f})")

        logging.info(f"   [3/4] Đang huấn luyện Asymmetric Fuzzy chính thức và theo dõi Overfitting...")
        final_model = AsymmetricFuzzyLagrangianMIL(C=best_C, max_iter=15, lr_lambda=0.01)
        final_model.fit(scaled_pca_train_bags, train_labels, eval_bags=scaled_pca_test_bags, eval_labels=test_labels)

        all_learning_curves_train.append(final_model.history['train_acc'])
        all_learning_curves_test.append(final_model.history['test_acc'])

        logging.info("   [4/4] Đang dự báo và tìm ngưỡng cắt tối ưu...")
        y_scores = final_model.predict(scaled_pca_test_bags)
        
        fpr, tpr, thresholds = roc_curve(test_labels, y_scores)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        y_pred = (y_scores >= optimal_threshold).astype(int)

        acc = accuracy_score(test_labels, y_pred)
        f1 = f1_score(test_labels, y_pred, zero_division=0)
        auc_score = roc_auc_score(test_labels, y_scores)       
        metrics['acc'].append(acc)
        metrics['f1'].append(f1)
        metrics['auc'].append(auc_score)
        
        all_true_labels.extend(test_labels)
        all_pred_labels.extend(y_pred)
        
        fpr_fold, tpr_fold, _ = roc_curve(test_labels, y_scores)
        interp_tpr = np.interp(mean_fpr, fpr_fold, tpr_fold)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc_score)
        
        ax_roc.plot(fpr_fold, tpr_fold, lw=1.5, alpha=0.4, 
                    label=f'ROC Fold {fold+1} (AUC = {auc_score:.4f})')
        
        fold_time = time.time() - fold_start_time
        logging.info("-" * 50)
        logging.info(f"-> FOLD {fold+1} KQ CHÍNH THỨC | C={best_C} | Threshold={optimal_threshold:.4f}")
        logging.info(f"-> Acc={acc:.4f} | F1={f1:.4f} | AUC={auc_score:.4f} | T.gian: {fold_time:.1f}s")
        logging.info("-" * 50)

    logging.info("\n" + "="*50)
    logging.info("KẾT QUẢ TRUNG BÌNH CỦA MÔ HÌNH SAU 5-FOLD CV:")
    logging.info(f"Accuracy : {np.mean(metrics['acc']):.4f}")
    logging.info(f"F1-Score : {np.mean(metrics['f1']):.4f}")
    logging.info(f"AUC-ROC  : {np.mean(metrics['auc']):.4f}")
    logging.info("="*50)

    # ==========================================
    # KẾT XUẤT VÀ HIỂN THỊ ĐỒ THỊ 
    # ==========================================
    from datetime import datetime
    current_time = datetime.now().strftime("%H%M%S")

    logging.info("\nĐang xuất biểu đồ ROC, Confusion Matrix và Learning Curve...")

    # --- 1. ĐỒ THỊ LEARNING CURVE ---
    mean_train_acc = np.mean(all_learning_curves_train, axis=0)
    mean_test_acc = np.mean(all_learning_curves_test, axis=0)
    iterations = range(1, 16) 

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, mean_train_acc, marker='o', color='#d62728', label='Accuracy trên tập Train', linewidth=2.5)
    plt.plot(iterations, mean_test_acc, marker='s', color='#1f77b4', label='Accuracy trên tập Test', linewidth=2.5)
    
    plt.title('Learning Curve', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Số vòng lặp tối ưu', fontsize=12, fontweight='bold')
    plt.ylabel('Độ chính xác (Accuracy)', fontsize=12, fontweight='bold')
    plt.xticks(iterations)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"AsymFuzzy_Learning_Curve_{current_time}.png", dpi=300)
    plt.show() 

    # --- 2. ĐƯỜNG CONG ROC ---
    ax_roc.plot([0, 1], [0, 1], linestyle='--', lw=2, color='red', label='Đoán ngẫu nhiên', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    
    ax_roc.plot(mean_fpr, mean_tpr, color='blue', 
                label=f'ROC trung bình (AUC = {mean_auc:.4f} $\pm$ {std_auc:.4f})', 
                lw=3, alpha=.9)

    ax_roc.set_xlim([-0.05, 1.05])
    ax_roc.set_ylim([-0.05, 1.05])
    ax_roc.set_xlabel('Tỷ lệ dương tính giả (FPR)', fontsize=12, fontweight='bold')
    ax_roc.set_ylabel('Tỷ lệ dương tính thật (TPR)', fontsize=12, fontweight='bold')
    ax_roc.set_title('Đường cong ROC qua 5-Fold', fontsize=14, fontweight='bold', pad=15)
    ax_roc.legend(loc="lower right", fontsize=10)
    ax_roc.grid(True, linestyle='--', alpha=0.6)
    
    fig_roc.tight_layout()
    fig_roc.savefig(f"AsymFuzzy_ROC_Curve_{current_time}.png", dpi=300) 
    plt.show() 

    # --- 3. MA TRẬN NHẦM LẪN ---
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(all_true_labels, all_pred_labels)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['LUSC (0)', 'LUAD (1)'],
                yticklabels=['LUSC (0)', 'LUAD (1)'],
                annot_kws={"size": 14, "weight": "bold"},
                cbar_kws={'label': 'Số lượng bệnh nhân'})

    plt.title('Ma trận nhầm lẫn tổng hợp', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Nhãn dự báo (Predicted Label)', fontsize=12, fontweight='bold')
    plt.ylabel('Nhãn thực tế (True Label)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"AsymFuzzy_Confusion_Matrix_{current_time}.png", dpi=300) 
    plt.show()

    logging.info("=> THÀNH CÔNG: Đã hoàn thành huấn luyện!")