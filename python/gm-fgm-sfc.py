import pandas as pd
import matplotlib.pyplot as plt

fgm_headers = ['id', 'threshold', 'batch-size', 'sites',
               'rounds', 'subrounds', 'accuracy', 'traffic']

fgm_df = pd.read_csv('../results/fgm-sfc.csv', usecols=fgm_headers)

gm_headers = ['id', 'threshold', 'batch-size', 'sites',
              'rounds', 'rebalances', 'accuracy', 'traffic']

gm_df = pd.read_csv('../results/gm-sfc.csv', usecols=gm_headers)

FONT_SIZE = 30
plt.rc('font', size=(FONT_SIZE - 8))


# #### Model accuracy and rounds for various **thresholds** ####
# - ***Threshold: { 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1 }***
# - Batch size: 16
# - Sites: 8
# 
# (Note! Experiment executes: 30) 
fgm_subdf = fgm_df[(fgm_df['id'] >= 1) & (fgm_df['id'] <= 7)]
_sorted_fgm = fgm_subdf.sort_values('threshold')

gm_subdf = gm_df[(gm_df['id'] >= 1) & (gm_df['id'] <= 7)]
_sorted_gm = gm_subdf.sort_values('threshold')

x = _sorted_fgm['threshold']

# Accuracy
y_fgm = _sorted_fgm['accuracy']
y_gm = _sorted_gm['accuracy']
y_centr = [99.59, 99.59, 99.59, 99.59, 99.59, 99.59, 99.59]
plt.figure(figsize=(16, 12))
plt.plot(x, y_fgm, label="fgm", marker='D', linewidth=4)
plt.plot(x, y_gm, label="gm", marker='D', linewidth=4)
plt.plot(x, y_centr, label="centralized", linewidth=4)
plt.xlabel('Threshold', fontsize=FONT_SIZE)
plt.ylabel('Accuracy', fontsize=FONT_SIZE)
plt.legend(loc='best', fontsize=FONT_SIZE)
plt.grid(True)
plt.xticks(x, fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.savefig('../results/sfc-plots/exp_Fig_1_1')

# Rounds
y_fgm = _sorted_fgm['rounds']
y_gm = _sorted_gm['rounds']
plt.figure(figsize=(16, 12))
plt.plot(x, y_fgm, label="fgm", marker='D', linewidth=4)
plt.plot(x, y_gm, label="gm", marker='D', linewidth=4)
plt.xlabel('Threshold', fontsize=FONT_SIZE)
plt.ylabel('Rounds', fontsize=FONT_SIZE)
plt.legend(loc='best', fontsize=FONT_SIZE)
plt.grid(True)
plt.xticks(x, fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.savefig("../results/sfc-plots/exp_Fig_1_2")

# Traffic
y_fgm = _sorted_fgm['traffic']
y_gm = _sorted_gm['traffic']
plt.figure(figsize=(16, 12))
plt.plot(x, y_fgm, label="fgm", marker='D', linewidth=4)
plt.plot(x, y_gm, label="gm", marker='D', linewidth=4)
plt.xlabel('Threshold', fontsize=FONT_SIZE)
plt.ylabel('Traffic', fontsize=FONT_SIZE)
plt.legend(loc='best', fontsize=FONT_SIZE)
plt.grid(True)
plt.xticks(x, fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.savefig("../results/sfc-plots/exp_Fig_1_3")

# #### Model accuracy and rounds for various **batch sizes** ####
# - Threshold: 0.5
# - ***Batch size: {1, 4, 16, 32, 64, 128}***
# - Sites: 8
#
# (Note! Experiment executes: 30)
fgm_subdf = fgm_df[(fgm_df['id'] >= 8) & (fgm_df['id'] <= 13)]
_sorted_fgm = fgm_subdf.sort_values('batch-size')

gm_subdf = gm_df[(gm_df['id'] >= 8) & (gm_df['id'] <= 13)]
_sorted_gm = gm_subdf.sort_values('batch-size')

x = _sorted_fgm['batch-size']

# Accuracy
y_fgm = _sorted_fgm['accuracy']
y_gm = _sorted_gm['accuracy']
y_centr = [99.59, 99.59, 99.59, 99.59, 99.59, 99.59]
plt.figure(figsize=(16, 12))
plt.plot(x, y_fgm, label="fgm", marker='D', linewidth=4)
plt.plot(x, y_gm, label="gm", marker='D', linewidth=4)
plt.plot(x, y_centr, label="centralized", linewidth=4)
plt.xlabel('Batch Size', fontsize=FONT_SIZE)
plt.ylabel('Accuracy', fontsize=FONT_SIZE)
plt.legend(loc='best', fontsize=FONT_SIZE)
plt.grid(True)
plt.xticks(x, fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.savefig("../results/sfc-plots/exp_Fig_2_1")

# Rounds
y_fgm = _sorted_fgm['rounds']
y_gm = _sorted_gm['rounds']
plt.figure(figsize=(16, 12))
plt.plot(x, y_fgm, label="fgm", marker='D', linewidth=4)
plt.plot(x, y_gm, label="gm", marker='D', linewidth=4)
plt.xlabel('Batch Size', fontsize=FONT_SIZE)
plt.ylabel('Rounds', fontsize=FONT_SIZE)
plt.legend(loc='best', fontsize=FONT_SIZE)
plt.grid(True)
plt.xticks(x, fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.savefig("../results/sfc-plots/exp_Fig_2_2")

# Traffic
y_fgm = _sorted_fgm['traffic']
y_gm = _sorted_gm['traffic']
plt.figure(figsize=(16, 12))
plt.plot(x, y_fgm, label="fgm", marker='D', linewidth=4)
plt.plot(x, y_gm, label="gm", marker='D', linewidth=4)
plt.xlabel('Batch size', fontsize=FONT_SIZE)
plt.ylabel('Traffic', fontsize=FONT_SIZE)
plt.legend(loc='best', fontsize=FONT_SIZE)
plt.grid(True)
plt.xticks(x, fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.savefig("../results/sfc-plots/exp_Fig_2_3")

# #### Model accuracy and rounds for various **sites** ####
# - Threshold: 0.5
# - Batch size: 16
# - ***Sites: {4, 8, 16, 32, 64, 128}***
#
# (Note! Experiment executes: 30)
fgm_subdf = fgm_df[(fgm_df['id'] >= 14) & (fgm_df['id'] <= 19)]
_sorted_fgm = fgm_subdf.sort_values('sites')

gm_subdf = gm_df[(gm_df['id'] >= 14) & (gm_df['id'] <= 19)]
_sorted_gm = gm_subdf.sort_values('sites')

x = _sorted_fgm['sites']

# Accuracy
y_fgm = _sorted_fgm['accuracy']
y_gm = _sorted_gm['accuracy']
y_centr = [99.59, 99.59, 99.59, 99.59, 99.59, 99.59]
plt.figure(figsize=(16, 12))
plt.plot(x, y_fgm, label="fgm", marker='D', linewidth=4)
plt.plot(x, y_gm, label="gm", marker='D', linewidth=4)
plt.plot(x, y_centr, label="centralized", linewidth=4)
plt.xlabel('Sites', fontsize=FONT_SIZE)
plt.ylabel('Accuracy', fontsize=FONT_SIZE)
plt.legend(loc='best', fontsize=FONT_SIZE)
plt.grid(True)
plt.xticks(x, fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.savefig("../results/sfc-plots/exp_Fig_3_1")

# Rounds
y_fgm = _sorted_fgm['rounds']
y_gm = _sorted_gm['rounds']
plt.figure(figsize=(16, 12))
plt.plot(x, y_fgm, label="fgm", marker='D', linewidth=4)
plt.plot(x, y_gm, label="gm", marker='D', linewidth=4)
plt.xlabel('Sites', fontsize=FONT_SIZE)
plt.ylabel('Rounds', fontsize=FONT_SIZE)
plt.legend(loc='best', fontsize=FONT_SIZE)
plt.grid(True)
plt.xticks(x, fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.savefig("../results/sfc-plots/exp_Fig_3_2")

# Traffic
y_fgm = _sorted_fgm['traffic']
y_gm = _sorted_gm['traffic']
plt.figure(figsize=(16, 12))
plt.plot(x, y_fgm, label="fgm", marker='D', linewidth=4)
plt.plot(x, y_gm, label="gm", marker='D', linewidth=4)
plt.xlabel('Sites', fontsize=FONT_SIZE)
plt.ylabel('Traffic', fontsize=FONT_SIZE)
plt.legend(loc='best', fontsize=FONT_SIZE)
plt.grid(True)
plt.xticks(x, fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.savefig("../results/sfc-plots/exp_Fig_3_3")
