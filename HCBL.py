import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import Counter
import plotly.graph_objects as go
from sklearn.metrics import (f1_score, accuracy_score, precision_score,
                             recall_score, confusion_matrix, classification_report,
                             balanced_accuracy_score, matthews_corrcoef,
                             cohen_kappa_score, roc_auc_score, log_loss)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import RepeatedStratifiedKFold
import tensorflow as tf
from tensorflow.keras.layers import (Input, Flatten, Dense, Dropout, Conv2D,
                                     MaxPooling2D, LSTM, TimeDistributed,
                                     Bidirectional, BatchNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.regularizers import l2

# ========================= 参数配置 =========================
way = 'ISD'
input_path = r"D:\研究\代码\代码\code"
output_path = rf"D:\研究\代码\代码\code\流体\CNN_BiLSTM\1"
plt.rcParams.update({'font.size': 18, 'font.family': 'Times New Roman'})

# 图像与模型参数
img_height, img_width, img_channels = 21, 15, 3
n_classes = 3
sequence_length = 5
n_folds, n_repeats = 3, 1
seed = 50
batch_size = 32
epochs = 200

np.random.seed(seed)
tf.random.set_seed(seed)

# 井位坐标
WELL_COORDS = {
    'Well-A': {'inline': 2467, 'xline': 3714},
    'Well-B': {'inline': 2589, 'xline': 3891},
    'Well-C': {'inline': 2391, 'xline': 4059}
}
WELL_CYLINDER_RADIUS = 6
TIME_START, TIME_INTERVAL, N_TIME_SAMPLES = 3200, 4, 100


# ========================= 可视化配置 =========================
class VisualizationConfig:
    CLASS_NAMES = {0: 'Gas', 1: 'Water', 2: 'Non-reservoir'}
    CLASS_COLORS = {0: '#E74C3C', 1: '#3498DB', 2: '#FFFFFF'}
    CLASS_COLORS_RGB = {
        0: (231 / 255, 76 / 255, 60 / 255),
        1: (52 / 255, 152 / 255, 219 / 255),
        2: (1.0, 1.0, 1.0)
    }
    AXIS_RANGES = {'inline': [2300, 2650], 'xline': [3650, 4100]}
    FIGURE_SIZE = {'width': 1800, 'height': 1200}

    # 128色色标
    CUSTOM_COLORSCALE_128 = [
        [0.0, 'rgb(0,0,159)'], [0.1, 'rgb(0,0,240)'], [0.2, 'rgb(0,94,255)'],
        [0.3, 'rgb(0,190,255)'], [0.4, 'rgb(52,255,202)'], [0.5, 'rgb(132,255,120)'],
        [0.6, 'rgb(250,249,2)'], [0.7, 'rgb(255,154,0)'], [0.8, 'rgb(255,42,0)'],
        [0.9, 'rgb(198,0,0)'], [1.0, 'rgb(128,0,0)']
    ]


# ========================= 工具函数 =========================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_images(image_folder):
    from PIL import Image
    images = []
    for f in sorted(os.listdir(image_folder)):
        if f.endswith('.bmp'):
            with Image.open(os.path.join(image_folder, f)) as img:
                arr = np.array(img, dtype=np.float32) / 255.0
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=-1)
                images.append(arr)
    return np.array(images)


def load_hard_labels(labels_file):
    df = pd.read_excel(labels_file, header=None)
    labels = df.iloc[:, 0].values.astype(np.int32)
    print(f"  ✅ 加载标签: {os.path.basename(labels_file)}, 分布: {Counter(labels)}")
    return labels


def convert_4class_to_3class(labels):
    new_labels = labels.copy()
    new_labels[new_labels == 3] = 2
    return new_labels


def hard_to_onehot(hard_labels, n_classes=3):
    onehot = np.zeros((len(hard_labels), n_classes), dtype=np.float32)
    for i, label in enumerate(hard_labels):
        onehot[i, label] = 1.0
    return onehot


# ========================= 序列数据生成 =========================
def create_sequences_preserve_wells(X, y_hard, y_onehot, well_lengths, seq_len=5):
    X_seq, y_hard_seq, y_onehot_seq = [], [], []
    start_idx, half_seq = 0, seq_len // 2

    for well_len in well_lengths:
        X_well = X[start_idx:start_idx + well_len]
        y_h, y_o = y_hard[start_idx:start_idx + well_len], y_onehot[start_idx:start_idx + well_len]
        X_pad = np.pad(X_well, ((half_seq, half_seq), (0, 0), (0, 0), (0, 0)), mode='edge')

        for i in range(well_len):
            X_seq.append(X_pad[i:i + seq_len])
            y_hard_seq.append(y_h[i])
            y_onehot_seq.append(y_o[i])
        start_idx += well_len

    return np.array(X_seq), np.array(y_hard_seq), np.array(y_onehot_seq)


def create_sequences_for_well(X, seq_len=5):
    half_seq = seq_len // 2
    X_pad = np.pad(X, ((half_seq, half_seq), (0, 0), (0, 0), (0, 0)), mode='edge')
    return np.array([X_pad[i:i + seq_len] for i in range(len(X))])


# ========================= CNN-BiLSTM模型 =========================
def cnn_lstm_model(img_h, img_w, img_c, n_cls, seq_len=5):
    inputs = Input(shape=(seq_len, img_h, img_w, img_c))

    # CNN特征提取
    x = TimeDistributed(Conv2D(32, (3, 3), padding="same", kernel_regularizer=l2(0.02)))(inputs)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(tf.keras.layers.Activation('relu'))(x)
    x = TimeDistributed(Conv2D(32, (3, 3), padding="same", kernel_regularizer=l2(0.02)))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(tf.keras.layers.Activation('relu'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(Dropout(0.2))(x)

    x = TimeDistributed(Conv2D(64, (3, 3), padding="same", kernel_regularizer=l2(0.02)))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(tf.keras.layers.Activation('relu'))(x)
    x = TimeDistributed(Conv2D(64, (3, 3), padding="same", kernel_regularizer=l2(0.02)))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(tf.keras.layers.Activation('relu'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(Dropout(0.3))(x)

    x = TimeDistributed(Conv2D(128, (3, 3), padding="same", kernel_regularizer=l2(0.02)))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(tf.keras.layers.Activation('relu'))(x)
    x = TimeDistributed(Dropout(0.3))(x)
    x = TimeDistributed(Flatten())(x)

    # BiLSTM
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))(x)
    x = Bidirectional(LSTM(32, return_sequences=False, dropout=0.3, recurrent_dropout=0.2))(x)

    # 全连接
    x = Dense(128, kernel_regularizer=l2(0.02))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(64, kernel_regularizer=l2(0.02))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Dropout(0.4)(x)

    outputs = Dense(n_cls, activation='softmax')(x)

    model = Model(inputs, outputs, name='CNN_BiLSTM')
    model.compile(loss=CategoricalCrossentropy(label_smoothing=0.0),
                  optimizer=Adam(learning_rate=0.0005), metrics=['accuracy'])
    return model


def create_model_for_fold():
    model = cnn_lstm_model(img_height, img_width, img_channels, n_classes, sequence_length)
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor='val_loss', patience=12, verbose=1, restore_best_weights=True)
    ]
    return model, callbacks


# ========================= 集成预测 =========================
def ensemble_predict(models, X, batch_size=32):
    predictions = None
    for i, model in enumerate(models):
        pred = model.predict(X, batch_size=batch_size, verbose=0)
        predictions = pred if predictions is None else predictions + pred
        print(f"    模型 {i + 1}/{len(models)} 预测完成")
    return predictions / len(models)


# ========================= 评估函数 =========================
def calculate_metrics(y_true, y_pred, classes):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    print("\n  📊 分类报告:")
    print(classification_report(y_true, y_pred, target_names=[classes[i] for i in range(len(classes))], digits=4))
    return metrics


def print_metrics_summary(metrics, name):
    print(f"\n  {'=' * 60}\n  📈 {name} - 性能指标\n  {'=' * 60}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(
        f"  Macro Precision/Recall/F1: {metrics['precision_macro']:.4f} / {metrics['recall_macro']:.4f} / {metrics['f1_macro']:.4f}")
    print(
        f"  Weighted Precision/Recall/F1: {metrics['precision_weighted']:.4f} / {metrics['recall_weighted']:.4f} / {metrics['f1_weighted']:.4f}")


def calculate_additional_metrics(y_true, y_proba, classes):
    y_pred = np.argmax(y_proba, axis=1)
    print(f"\n  {'=' * 60}\n  📊 额外评估指标\n  {'=' * 60}")
    print(f"  平衡准确率: {balanced_accuracy_score(y_true, y_pred):.4f}")
    print(f"  MCC: {matthews_corrcoef(y_true, y_pred):.4f}")
    print(f"  Kappa: {cohen_kappa_score(y_true, y_pred):.4f}")
    try:
        print(f"  AUC (OvR): {roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro'):.4f}")
    except:
        pass
    print(f"  Log Loss: {log_loss(y_true, y_proba):.4f}")


def plot_confusion_matrix(y_true, y_pred, classes, output_path, name, normalize=False):
    ensure_dir(output_path)
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap='Blues')
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)

    labels = ['Gas', 'Water', 'Non-\nreservoir']
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_yticklabels(labels, fontsize=14)
    ax.set_xlabel('Predicted', fontsize=16, weight='bold')
    ax.set_ylabel('True', fontsize=16, weight='bold')

    thresh = cm.max() / 2.
    for i in range(len(classes)):
        for j in range(len(classes)):
            fmt = f'{cm[i, j]:.2f}' if normalize else f'{cm[i, j]:.0f}'
            ax.text(j, i, fmt, ha='center', va='center', fontsize=18, weight='bold',
                    color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    suffix = '_normalized' if normalize else ''
    plt.savefig(os.path.join(output_path, f'{name}_CM{suffix}.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_probability_bars(probs, classes, colors, out_dir, fname):
    ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(3, 10))
    left = np.zeros(len(probs))
    for c in range(probs.shape[1]):
        ax.barh(range(len(probs)), probs[:, c], left=left, color=colors[c], height=1.0)
        left += probs[:, c]
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(probs) - 0.5)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), bbox_inches='tight', dpi=600)
    plt.close()


def plot_lithology(labels, out_dir, colors, fname="lithology.png"):
    ensure_dir(out_dir)
    img = np.array([[mcolors.to_rgb(colors[int(l)])] for l in labels])
    plt.figure(figsize=(2, 10))
    plt.imshow(img, aspect='auto')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), bbox_inches='tight', dpi=600)
    plt.close()


# ========================= 剖面处理类 =========================
class CrossSectionProcessor:
    def __init__(self, well_coords, time_start=3200, time_interval=4, n_time=100):
        self.well_coords = well_coords
        self.time_start, self.time_interval = time_start, time_interval
        self.time_samples = np.arange(time_start, time_start + n_time * time_interval, time_interval)
        self.all_inlines = np.arange(2327, 2628, 2)
        self.all_xlines = np.arange(3678, 4079, 2)

    def calculate_cross_section_points(self, well1, well2):
        c1, c2 = self.well_coords[well1], self.well_coords[well2]
        i1, x1, i2, x2 = c1['inline'], c1['xline'], c2['inline'], c2['xline']
        di, dx = i2 - i1, x2 - x1
        length = np.sqrt(di ** 2 + dx ** 2)
        if length < 1e-6: return [(i1, x1)]

        di_n, dx_n = di / length, dx / length
        il_min, il_max = self.all_inlines.min(), self.all_inlines.max()
        xl_min, xl_max = self.all_xlines.min(), self.all_xlines.max()

        t_cand = []
        if abs(di_n) > 1e-9: t_cand.extend([(il_min - i1) / di_n, (il_max - i1) / di_n])
        if abs(dx_n) > 1e-9: t_cand.extend([(xl_min - x1) / dx_n, (xl_max - x1) / dx_n])

        valid_t = [t for t in t_cand if
                   il_min - 1 <= i1 + t * di_n <= il_max + 1 and xl_min - 1 <= x1 + t * dx_n <= xl_max + 1]
        t_min, t_max = (min(valid_t), max(valid_t)) if len(valid_t) >= 2 else (0, length)

        t_vals = np.linspace(t_min, t_max, max(int((t_max - t_min) * 3), 1000))
        il_coords, xl_coords = i1 + t_vals * di_n, x1 + t_vals * dx_n
        mask = (il_min <= il_coords) & (il_coords <= il_max) & (xl_min <= xl_coords) & (xl_coords <= xl_max)

        il_idx = np.clip(np.searchsorted(self.all_inlines, il_coords[mask]), 0, len(self.all_inlines) - 1)
        xl_idx = np.clip(np.searchsorted(self.all_xlines, xl_coords[mask]), 0, len(self.all_xlines) - 1)

        return sorted(set(zip(self.all_inlines[il_idx], self.all_xlines[xl_idx])))

    def load_seismic_images(self, folder, locations):
        from PIL import Image
        file_map = {f"ISD_{i:04d}.bmp": self.time_start + (i - 1) * self.time_interval for i in
                    range(1, N_TIME_SAMPLES + 1)}

        all_data = []
        for il, xl in sorted(locations):
            path = os.path.join(folder, f"Inline_{il}_Xline_{xl}")
            if os.path.exists(path):
                for fname, t in file_map.items():
                    fpath = os.path.join(path, fname)
                    if os.path.exists(fpath):
                        all_data.append((fpath, il, xl, t))

        if not all_data: return np.array([]), []
        all_data.sort(key=lambda x: (x[1], x[2], x[3]))

        images, metadata = [], []
        for fpath, il, xl, t in all_data:
            try:
                with Image.open(fpath) as img:
                    arr = np.array(img, dtype=np.float32) / 255.0
                    if arr.ndim == 2:
                        arr = np.stack([arr] * 3, axis=-1)
                    elif arr.shape[2] == 4:
                        arr = arr[:, :, :3]
                    images.append(arr)
                    metadata.append((il, xl, t))
            except:
                pass

        print(f"  ✅ 加载 {len(images):,} 张图像")
        return np.array(images), metadata

    def create_section_data(self, pred_df, well1, well2):
        c1, c2 = self.well_coords[well1], self.well_coords[well2]
        di, dx = c2['inline'] - c1['inline'], c2['xline'] - c1['xline']

        pts = pred_df[['inline', 'xline']].drop_duplicates().copy()
        pts['proj'] = ((pts['inline'] - c1['inline']) * di + (pts['xline'] - c1['xline']) * dx) / (di ** 2 + dx ** 2)
        pts = pts.sort_values('proj')

        ils, xls = pts['inline'].values, pts['xline'].values
        n_tr, n_t = len(ils), len(self.time_samples)

        trace_map = {(il, xl): i for i, (il, xl) in enumerate(zip(ils, xls))}
        time_map = {t: i for i, t in enumerate(self.time_samples)}

        probs = {k: np.full((n_t, n_tr), np.nan, dtype=np.float32) for k in ['gas', 'water', 'mudstone']}
        for _, r in pred_df.iterrows():
            key = (r['inline'], r['xline'])
            if key in trace_map and r['time'] in time_map:
                ti, tri = time_map[r['time']], trace_map[key]
                probs['gas'][ti, tri], probs['water'][ti, tri], probs['mudstone'][ti, tri] = r['prob_gas'], r[
                    'prob_water'], r['prob_mudstone']

        well_pos = {}
        for wn in [well1, well2]:
            wc = self.well_coords[wn]
            dist = np.sqrt((ils - wc['inline']) ** 2 + (xls - wc['xline']) ** 2)
            idx = np.argmin(dist)
            well_pos[wn] = {'trace_idx': idx, 'inline': ils[idx], 'xline': xls[idx]}

        return {'inlines': ils, 'xlines': xls, 'time_samples': self.time_samples,
                'probabilities': probs, 'n_traces': n_tr, 'n_time': n_t, 'well_positions': well_pos}


# ========================= 剖面可视化类 =========================
class CrossSectionVisualizer:
    def __init__(self, config=None):
        self.config = config or VisualizationConfig()

    def create_probability_heatmap(self, sec_ab, sec_bc, class_name, out_path, well_labels=None):
        if class_name.lower() not in ['gas', 'water']: return
        print(f"\n【创建 {class_name} 概率热力图】")

        fig = go.Figure()
        colorscale = self.config.CUSTOM_COLORSCALE_128

        for sec, name, show_cb in [(sec_ab, 'AB', True), (sec_bc, 'BC', False)]:
            prob = sec['probabilities'][class_name.lower()]
            X, Z = np.meshgrid(sec['inlines'], sec['time_samples'])
            Y, _ = np.meshgrid(sec['xlines'], sec['time_samples'])

            fig.add_trace(go.Surface(
                x=X, y=Y, z=Z, surfacecolor=prob, colorscale=colorscale, cmin=0, cmax=1,
                showscale=show_cb, name=f'{name} Section', opacity=1.0,
                lighting=dict(ambient=0.9, diffuse=0.1, specular=0.0),
                colorbar=dict(title='Probability', tickvals=[0, 1], ticktext=['0', '1'], len=0.5) if show_cb else None
            ))

        if well_labels:
            self._add_well_cylinders(fig, sec_ab, sec_bc, well_labels)

        self._update_layout(fig, f'{class_name.capitalize()} Probability')
        ensure_dir(out_path)
        fig.write_html(os.path.join(out_path, f'CrossSection_{class_name}_Prob.html'))
        try:
            fig.write_image(os.path.join(out_path, f'CrossSection_{class_name}_Prob.png'), width=1800, height=1200,
                            scale=2)
        except:
            pass

    def create_classification_map(self, sec_ab, sec_bc, well_labels, out_path):
        print(f"\n【创建分类结果图】")
        fig = go.Figure()

        discrete_cs = [[0, self.config.CLASS_COLORS[0]], [0.33, self.config.CLASS_COLORS[0]],
                       [0.33, self.config.CLASS_COLORS[1]], [0.67, self.config.CLASS_COLORS[1]],
                       [0.67, self.config.CLASS_COLORS[2]], [1.0, self.config.CLASS_COLORS[2]]]

        for sec, name in [(sec_ab, 'AB'), (sec_bc, 'BC')]:
            probs = np.stack([sec['probabilities'][k] for k in ['gas', 'water', 'mudstone']], axis=-1)
            cls = np.argmax(probs, axis=-1).astype(float)
            cls[np.isnan(sec['probabilities']['gas'])] = np.nan

            X, Z = np.meshgrid(sec['inlines'], sec['time_samples'])
            Y, _ = np.meshgrid(sec['xlines'], sec['time_samples'])

            fig.add_trace(go.Surface(
                x=X, y=Y, z=Z, surfacecolor=cls, colorscale=discrete_cs, cmin=0, cmax=2,
                showscale=False, name=f'{name} Section', opacity=1.0,
                lighting=dict(ambient=0.9, diffuse=0.1, specular=0.0)
            ))

        self._add_well_cylinders(fig, sec_ab, sec_bc, well_labels)

        for ci, cn in self.config.CLASS_NAMES.items():
            fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers',
                                       marker=dict(size=15, color=self.config.CLASS_COLORS[ci]), name=cn,
                                       showlegend=True))

        self._update_layout(fig, 'Fluid Classification')
        ensure_dir(out_path)
        fig.write_html(os.path.join(out_path, 'CrossSection_Classification.html'))
        try:
            fig.write_image(os.path.join(out_path, 'CrossSection_Classification.png'), width=1800, height=1200, scale=2)
        except:
            pass

    def _add_well_cylinders(self, fig, sec_ab, sec_bc, well_labels):
        radius, n_theta = WELL_CYLINDER_RADIUS, 20
        for wn in ['Well-A', 'Well-B', 'Well-C']:
            if wn not in well_labels: continue
            times, vals = well_labels[wn]
            if times is None: continue
            wc = WELL_COORDS[wn]

            for ci in [0, 1, 2]:
                mask = (vals == ci)
                if not np.any(mask): continue
                segs = self._get_segments(times[mask], TIME_INTERVAL)
                for t0, t1 in segs:
                    self._add_cylinder(fig, wc['inline'], wc['xline'], t0 - 2, t1 + 2, radius,
                                       self.config.CLASS_COLORS[ci], n_theta)

    def _get_segments(self, times, interval):
        if len(times) == 0: return []
        times = np.sort(times)
        segs, start, end = [], times[0], times[0]
        for t in times[1:]:
            if t - end <= interval + 0.1:
                end = t
            else:
                segs.append((start, end)); start = end = t
        segs.append((start, end))
        return segs

    def _add_cylinder(self, fig, cx, cy, z0, z1, r, color, n):
        theta = np.linspace(0, 2 * np.pi, n + 1)
        z_vals = np.array([z0, z1])
        theta_g, z_g = np.meshgrid(theta, z_vals)
        x_g, y_g = cx + r * np.cos(theta_g), cy + r * np.sin(theta_g)

        fig.add_trace(go.Surface(x=x_g, y=y_g, z=z_g, surfacecolor=np.ones_like(z_g),
                                 colorscale=[[0, color], [1, color]], showscale=False, opacity=1.0, showlegend=False))

    def _update_layout(self, fig, title):
        fig.update_layout(
            title=dict(text=f'Cross Sections - {title}', font=dict(size=22)),
            scene=dict(
                xaxis=dict(title='Inline', range=self.config.AXIS_RANGES['inline']),
                yaxis=dict(title='Xline', range=self.config.AXIS_RANGES['xline'], tickvals=[3800, 3900, 4000, 4100]),
                zaxis=dict(title='Time (ms)', autorange='reversed'),
                aspectmode='manual', aspectratio=dict(x=1.5, y=1.5, z=1),
                camera=dict(eye=dict(x=-1.5, y=-1.5, z=0.8))
            ),
            width=1800, height=1200, margin=dict(l=50, r=150, t=80, b=50)
        )


# ========================= 剖面预测 =========================
def predict_section(X_imgs, metadata, models, processor, seq_len):
    loc_dict = {}
    for i, (il, xl, t) in enumerate(metadata):
        loc_dict.setdefault((il, xl), []).append((i, t))

    X_seq, meta_seq = [], []
    half = seq_len // 2

    for (il, xl), items in sorted(loc_dict.items()):
        items.sort(key=lambda x: x[1])
        idx, times = [x[0] for x in items], [x[1] for x in items]
        imgs = X_imgs[idx]
        padded = np.pad(imgs, ((half, half), (0, 0), (0, 0), (0, 0)), mode='edge')
        for i, t in enumerate(times):
            X_seq.append(padded[i:i + seq_len])
            meta_seq.append((il, xl, t))

    X_seq = np.array(X_seq)
    proba = ensemble_predict(models, X_seq, batch_size=64)

    return pd.DataFrame({
        'inline': [m[0] for m in meta_seq], 'xline': [m[1] for m in meta_seq], 'time': [m[2] for m in meta_seq],
        'prob_gas': proba[:, 0], 'prob_water': proba[:, 1], 'prob_mudstone': proba[:, 2]
    })


def load_well_labels_3class(well_name):
    mapping = {'Well-A': '1', 'Well-B': '2', 'Well-C': '3'}
    fpath = rf'C:\Users\Windows10\Desktop\isd\标签\Fulide_lables\标签\{mapping[well_name]}.xlsx'
    if not os.path.exists(fpath): return None, None

    try:
        labels = pd.read_excel(fpath, header=None).iloc[:, 0].values
        if mapping[well_name] in ['1', '3']:
            labels = labels[10:110]
        elif mapping[well_name] == '2':
            labels = labels[:-5]
        labels = convert_4class_to_3class(labels)
        times = np.arange(TIME_START, TIME_START + len(labels) * TIME_INTERVAL, TIME_INTERVAL)
        return times.astype(float), labels.astype(float)
    except:
        return None, None


def predict_and_visualize(seismic_folder, models, out_path, seq_len=5):
    print("\n" + "=" * 80 + "\n🎨 交叉剖面预测与可视化\n" + "=" * 80)

    processor = CrossSectionProcessor(WELL_COORDS, TIME_START, TIME_INTERVAL, N_TIME_SAMPLES)
    visualizer = CrossSectionVisualizer()

    well_labels = {wn: load_well_labels_3class(wn) for wn in ['Well-A', 'Well-B', 'Well-C']}
    well_labels = {k: v for k, v in well_labels.items() if v[0] is not None}

    sections = {}
    for (w1, w2), name in [(('Well-A', 'Well-B'), 'AB'), (('Well-B', 'Well-C'), 'BC')]:
        print(f"\n【处理{name}剖面】")
        pts = processor.calculate_cross_section_points(w1, w2)
        X, meta = processor.load_seismic_images(seismic_folder, set(pts))
        if len(X) == 0: continue

        pred = predict_section(X, meta, models, processor, seq_len)
        sections[name] = processor.create_section_data(pred, w1, w2)
        pred.to_csv(os.path.join(out_path, f"{name}_Predictions.csv"), index=False)

    if len(sections) == 2:
        vis_path = os.path.join(out_path, "Visualizations")
        for cn in ['gas', 'water']:
            visualizer.create_probability_heatmap(sections['AB'], sections['BC'], cn, vis_path, well_labels)
        visualizer.create_classification_map(sections['AB'], sections['BC'], well_labels, vis_path)

    print("\n  ✅ 可视化完成!")


# ========================= 主程序 =========================
if __name__ == "__main__":
    print("\n" + "=" * 80 + "\n🚀 CNN-BiLSTM 储层流体识别模型\n" + "=" * 80)

    # 数据路径
    sample_length = '21'
    folders = [rf'{input_path}\training and testing dieqian set\all\滑窗{w}\{way}_block_{sample_length}_{p}'
               for w in [1, 2, 3] for p in ['center', 'left', 'right']]
    label_files = [rf'C:\Users\Windows10\Desktop\isd\标签\Fulide_lables\标签\{i}.xlsx' for i in [1, 2, 3]]

    # 加载数据
    print("\n📂 加载数据...")
    X = [load_images(f) for f in folders]
    y_hard = [load_hard_labels(f) for f in label_files]

    # 调整数据
    y_hard[0], y_hard[2] = y_hard[0][10:110], y_hard[2][11:111]
    y_hard[1] = y_hard[1][:-4]
    X[3], X[4], X[5] = X[3][2:], X[4][2:], X[5][2:]

    # 合并数据
    X_train = np.concatenate([X[1], X[2], X[4], X[5], X[7], X[8]], axis=0)
    y_train_hard = np.concatenate([y_hard[0]] * 2 + [y_hard[1][:-1]] * 2 + [y_hard[2]] * 2, axis=0)
    X_test = np.concatenate([X[0], X[3], X[6]], axis=0)
    y_test_hard = np.concatenate([y_hard[0], y_hard[1][:-1], y_hard[2]], axis=0)

    train_well_lengths = [len(X[1]), len(X[2]), len(X[4]), len(X[5]), len(X[7]), len(X[8])]
    test_well_lengths = [len(X[0]), len(X[3]), len(X[6])]

    # 转换为3分类
    y_train_hard = convert_4class_to_3class(y_train_hard)
    y_test_hard = convert_4class_to_3class(y_test_hard)
    y_train_onehot = hard_to_onehot(y_train_hard, n_classes)
    y_test_onehot = hard_to_onehot(y_test_hard, n_classes)

    print(f"\n训练集: {X_train.shape}, 类别分布: {Counter(y_train_hard)}")
    print(f"测试集: {X_test.shape}, 类别分布: {Counter(y_test_hard)}")

    # 创建序列
    X_train_seq, y_train_hard_seq, y_train_onehot_seq = create_sequences_preserve_wells(
        X_train, y_train_hard, y_train_onehot, train_well_lengths, sequence_length)
    X_test_seq, y_test_hard_seq, y_test_onehot_seq = create_sequences_preserve_wells(
        X_test, y_test_hard, y_test_onehot, test_well_lengths, sequence_length)

    print(f"\n序列化后 - 训练: {X_train_seq.shape}, 测试: {X_test_seq.shape}")

    classes = {0: 'Gas', 1: 'Water', 2: 'Non-reservoir'}

    # 类别权重
    cw = compute_class_weight('balanced', classes=np.unique(y_train_hard_seq), y=y_train_hard_seq)
    class_weight_dict = {i: cw[i] * [0.7, 0.5, 1.0][i] for i in range(n_classes)}
    print(f"类别权重: {class_weight_dict}")

    # 交叉验证训练
    print("\n" + "=" * 80 + "\n🔄 交叉验证训练\n" + "=" * 80)
    rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=seed)
    fold_models, fold_scores = [], []

    for fold_idx, (tr_idx, val_idx) in enumerate(rskf.split(X_train_seq, y_train_hard_seq)):
        print(f"\n{'=' * 60}\n📁 Fold {fold_idx + 1}/{n_folds * n_repeats}\n{'=' * 60}")

        X_tr, X_val = X_train_seq[tr_idx], X_train_seq[val_idx]
        y_tr, y_val = y_train_onehot_seq[tr_idx], y_train_onehot_seq[val_idx]
        y_val_hard = y_train_hard_seq[val_idx]

        model, callbacks = create_model_for_fold()
        if fold_idx == 0: model.summary()

        model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=epochs,
                  batch_size=batch_size, callbacks=callbacks, class_weight=class_weight_dict, verbose=1)

        val_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
        acc, f1 = accuracy_score(y_val_hard, val_pred), f1_score(y_val_hard, val_pred, average='macro')
        print(f"\n  Fold {fold_idx + 1}: Acc={acc:.4f}, F1={f1:.4f}")

        fold_models.append(model)
        fold_scores.append({'accuracy': acc, 'f1': f1})

    # CV结果
    print(f"\n{'=' * 60}\n📊 CV结果\n{'=' * 60}")
    print(
        f"  Accuracy: {np.mean([s['accuracy'] for s in fold_scores]):.4f} ± {np.std([s['accuracy'] for s in fold_scores]):.4f}")
    print(f"  F1: {np.mean([s['f1'] for s in fold_scores]):.4f} ± {np.std([s['f1'] for s in fold_scores]):.4f}")

    # 集成评估
    print("\n" + "=" * 80 + "\n📊 集成模型评估\n" + "=" * 80)
    ensure_dir(output_path)

    # 训练集
    train_proba = ensemble_predict(fold_models, X_train_seq)
    train_pred = np.argmax(train_proba, axis=1)
    train_metrics = calculate_metrics(y_train_hard_seq, train_pred, classes)
    print_metrics_summary(train_metrics, "训练集")
    plot_confusion_matrix(y_train_hard_seq, train_pred, classes, output_path, "Train", False)
    plot_confusion_matrix(y_train_hard_seq, train_pred, classes, output_path, "Train", True)

    # 测试集
    test_proba = ensemble_predict(fold_models, X_test_seq)
    test_pred = np.argmax(test_proba, axis=1)  # 直接取概率最大值
    test_metrics = calculate_metrics(y_test_hard_seq, test_pred, classes)
    print_metrics_summary(test_metrics, "测试集")
    calculate_additional_metrics(y_test_hard_seq, test_proba, classes)
    plot_confusion_matrix(y_test_hard_seq, test_pred, classes, output_path, "Test", False)
    plot_confusion_matrix(y_test_hard_seq, test_pred, classes, output_path, "Test", True)

    # 保存指标
    pd.DataFrame({
        'Dataset': ['Training', 'Testing'],
        'Accuracy': [train_metrics['accuracy'], test_metrics['accuracy']],
        'F1_Macro': [train_metrics['f1_macro'], test_metrics['f1_macro']]
    }).to_csv(os.path.join(output_path, "Metrics.csv"), index=False)

    # 单井预测
    print("\n" + "=" * 80 + "\n🏭 单井预测\n" + "=" * 80)
    colors = VisualizationConfig.CLASS_COLORS_RGB

    for i, (X_w, name) in enumerate([(X[0], "Well1"), (X[3], "Well2"), (X[6], "Well3")]):
        print(f"\n【{name}预测】")
        X_seq = create_sequences_for_well(X_w, sequence_length)
        proba = ensemble_predict(fold_models, X_seq)
        pred = np.argmax(proba, axis=1)
        plot_probability_bars(proba, classes, colors, output_path, f"{name}_probs.png")
        plot_lithology(pred, output_path, colors, f"{name}_pred.png")

    # 3D剖面预测
    seismic_folder = rf'{input_path}\training and testing dieqian set\all\3D'
    if os.path.exists(seismic_folder):
        predict_and_visualize(seismic_folder, fold_models, output_path, sequence_length)

    # 保存模型
    print("\n" + "=" * 80 + "\n💾 保存模型\n" + "=" * 80)
    for i, m in enumerate(fold_models):
        m.save(os.path.join(output_path, f"model_fold_{i + 1}.keras"))
        print(f"  ✅ Fold {i + 1} 已保存")

    print("\n" + "=" * 80 + "\n🎉 完成!\n" + "=" * 80)