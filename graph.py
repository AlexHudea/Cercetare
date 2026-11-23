import matplotlib.pyplot as plt
import numpy as np

# Datele din rezultatele experimentale
scenarii = ['DAWN\nFog', 'DAWN\nRain', 'DAWN\nSand', 'DAWN\nSnow', 'Cityscapes\nClear']

yolo_map = [0.2894, 0.6337, 0.4172, 0.4420, 0.3688]
rcnn_map = [0.4427, 0.6614, 0.2368, 0.4460, 0.6997]

yolo_fps = [12.11, 11.62, 10.51, 10.90, 14.12]
rcnn_fps = [0.54, 0.50, 0.42, 0.44, 0.50]

x = np.arange(len(scenarii))
width = 0.36

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Comparație Performanță: YOLOv8s vs. Faster R-CNN', fontsize=15, fontweight='bold')

# --- Grafic 1: Acuratețe (mAP@50) ---
rects1 = ax1.bar(x - width/2, yolo_map, width, label='YOLOv8s', color='#2b8cbe')
rects2 = ax1.bar(x + width/2, rcnn_map, width, label='Faster R-CNN', color='#e34a33')

ax1.set_ylabel('mAP@50', fontsize=12)
ax1.set_title('Acuratețea Detecției (mAP@50)', fontsize=13)
ax1.set_xticks(x)
ax1.set_xticklabels(scenarii)
ax1.set_ylim(0, 0.85)
ax1.grid(axis='y', linestyle='--', alpha=0.6)
ax1.legend()

# --- Grafic 2: Viteză (FPS) ---
rects3 = ax2.bar(x - width/2, yolo_fps, width, label='YOLOv8s', color='#2b8cbe')
rects4 = ax2.bar(x + width/2, rcnn_fps, width, label='Faster R-CNN', color='#e34a33')

ax2.set_ylabel('Cadre pe secundă (FPS)', fontsize=12)
ax2.set_title('Viteza de Procesare (FPS - CPU)', fontsize=13)
ax2.set_xticks(x)
ax2.set_xticklabels(scenarii)
ax2.set_ylim(0, 17)
ax2.grid(axis='y', linestyle='--', alpha=0.6)
ax2.legend()

# Funcție pentru afișarea valorilor deasupra barelor
def adauga_etichete(ax, rects, format_str):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(format_str.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

adauga_etichete(ax1, rects1, '{:.3f}')
adauga_etichete(ax1, rects2, '{:.3f}')
adauga_etichete(ax2, rects3, '{:.1f}')
adauga_etichete(ax2, rects4, '{:.2f}')

plt.tight_layout()
plt.savefig('comparatie_yolo_rcnn.png', dpi=300)
plt.show()