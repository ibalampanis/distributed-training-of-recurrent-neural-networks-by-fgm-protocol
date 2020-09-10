import pandas as pd
import matplotlib.pyplot as plt

FONT_SIZE = 28
plt.rc('font', size=(FONT_SIZE - 8))
plt.rc('figure', max_open_warning=0)

fgm_headers = ['id', 'threshold', 'batch-size', 'sites', 'rounds', 'subrounds', 'accuracy', 'traffic']
gm_headers = ['id', 'threshold', 'batch-size', 'sites', 'rounds', 'rebalances', 'accuracy', 'traffic']

sf1_fgm_df = pd.read_csv('../results/SF1_fgm-sfc.csv', usecols=fgm_headers)
sf2_fgm_df = pd.read_csv('../results/SF2_fgm-sfc.csv', usecols=fgm_headers)
sf1_gm_df = pd.read_csv('../results/SF1_gm-sfc.csv', usecols=gm_headers)
sf2_gm_df = pd.read_csv('../results/SF2_gm-sfc.csv', usecols=gm_headers)

# Comparison between protocols using the 2nd type of safe function (spherical cap)

# San Fransisco Crime Classification Dataset

# Model accuracy and rounds for various thresholds
# - Threshold: { 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1 }
# - Batch size: 16
# - Workers: 8
fgm_subdf = sf2_fgm_df[(sf2_fgm_df['id'] >= 1) & (sf2_fgm_df['id'] <= 7)]
_sorted_fgm = fgm_subdf.sort_values('threshold')
gm_subdf = sf2_gm_df[(sf2_gm_df['id'] >= 1) & (sf2_gm_df['id'] <= 7)]
_sorted_gm = gm_subdf.sort_values('threshold')

x = _sorted_fgm['threshold']

# Accuracy
y_fgm = _sorted_fgm['accuracy']
y_gm = _sorted_gm['accuracy']
y_centr = [99.59, 99.59, 99.59, 99.59, 99.59, 99.59, 99.59]
plt.figure(figsize=(16, 12))
plt.plot(x, y_fgm, label="FGM", marker='D', linewidth=4)
plt.plot(x, y_gm, label="GM", marker='D', linewidth=4)
plt.plot(x, y_centr, label="Centralized", linewidth=4)
plt.xlabel('Threshold', fontsize=FONT_SIZE)
plt.ylabel('Accuracy', fontsize=FONT_SIZE)
plt.legend(loc='best', fontsize=FONT_SIZE)
plt.grid(True)
plt.title('SFCC Dataset')
plt.xticks(x, fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.savefig('../results/sfc-plots/exp_Fig_1_1')

# Rounds
y_fgm = _sorted_fgm['rounds']
y_gm = _sorted_gm['rounds']
plt.figure(figsize=(16, 12))
plt.plot(x, y_fgm, label="FGM", marker='D', linewidth=4)
plt.plot(x, y_gm, label="GM", marker='D', linewidth=4)
plt.xlabel('Threshold', fontsize=FONT_SIZE)
plt.ylabel('Rounds', fontsize=FONT_SIZE)
plt.legend(loc='best', fontsize=FONT_SIZE)
plt.grid(True)
plt.title('SFCC Dataset')
plt.xticks(x, fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.savefig('../results/sfc-plots/exp_Fig_1_2')

# Traffic
y_fgm = _sorted_fgm['traffic']
y_gm = _sorted_gm['traffic']
plt.figure(figsize=(16, 12))
plt.plot(x, y_fgm, label="FGM", marker='D', linewidth=4)
plt.plot(x, y_gm, label="GM", marker='D', linewidth=4)
plt.xlabel('Threshold', fontsize=FONT_SIZE)
plt.ylabel('Traffic', fontsize=FONT_SIZE)
plt.legend(loc='best', fontsize=FONT_SIZE)
plt.grid(True)
plt.title('SFCC Dataset')
plt.xticks(x, fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.savefig('../results/sfc-plots/exp_Fig_1_3')

# Model accuracy and rounds for various batch sizes
# - Threshold: 0.5
# - Batch size: {1, 4, 16, 32, 64, 128}
# - Workers: 8
fgm_subdf = sf2_fgm_df[(sf2_fgm_df['id'] >= 8) & (sf2_fgm_df['id'] <= 13)]
_sorted_fgm = fgm_subdf.sort_values('batch-size')
gm_subdf = sf2_gm_df[(sf2_gm_df['id'] >= 8) & (sf2_gm_df['id'] <= 13)]
_sorted_gm = gm_subdf.sort_values('batch-size')

x = _sorted_fgm['batch-size']

# Accuracy
y_fgm = _sorted_fgm['accuracy']
y_gm = _sorted_gm['accuracy']
y_centr = [99.59, 99.59, 99.59, 99.59, 99.59, 99.59]
plt.figure(figsize=(16, 12))
plt.plot(x, y_fgm, label="FGM", marker='D', linewidth=4)
plt.plot(x, y_gm, label="GM", marker='D', linewidth=4)
plt.plot(x, y_centr, label="Centralized", linewidth=4)
plt.xlabel('Batch Size', fontsize=FONT_SIZE)
plt.ylabel('Accuracy', fontsize=FONT_SIZE)
plt.legend(loc='best', fontsize=FONT_SIZE)
plt.grid(True)
plt.title('SFCC Dataset')
plt.xticks(x, fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.savefig('../results/sfc-plots/exp_Fig_2_1')

# Rounds
y_fgm = _sorted_fgm['rounds']
y_gm = _sorted_gm['rounds']
plt.figure(figsize=(16, 12))
plt.plot(x, y_fgm, label="FGM", marker='D', linewidth=4)
plt.plot(x, y_gm, label="GM", marker='D', linewidth=4)
plt.xlabel('Batch Size', fontsize=FONT_SIZE)
plt.ylabel('Rounds', fontsize=FONT_SIZE)
plt.legend(loc='best', fontsize=FONT_SIZE)
plt.grid(True)
plt.title('SFCC Dataset')
plt.xticks(x, fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.savefig('../results/sfc-plots/exp_Fig_2_2')

# Traffic
y_fgm = _sorted_fgm['traffic']
y_gm = _sorted_gm['traffic']
plt.figure(figsize=(16, 12))
plt.plot(x, y_fgm, label="FGM", marker='D', linewidth=4)
plt.plot(x, y_gm, label="GM", marker='D', linewidth=4)
plt.xlabel('Batch Size', fontsize=FONT_SIZE)
plt.ylabel('Traffic', fontsize=FONT_SIZE)
plt.legend(loc='best', fontsize=FONT_SIZE)
plt.grid(True)
plt.title('SFCC Dataset')
plt.xticks(x, fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.savefig('../results/sfc-plots/exp_Fig_2_3')

# Model accuracy and rounds for various sites
# - Threshold: 0.5
# - Batch size: 16
# - Workers: {4, 8, 16, 32, 64, 128}
fgm_subdf = sf2_fgm_df[(sf2_fgm_df['id'] >= 14) & (sf2_fgm_df['id'] <= 19)]
_sorted_fgm = fgm_subdf.sort_values('sites')
gm_subdf = sf2_gm_df[(sf2_gm_df['id'] >= 14) & (sf2_gm_df['id'] <= 19)]
_sorted_gm = gm_subdf.sort_values('sites')

x = _sorted_fgm['sites']

# Accuracy
y_fgm = _sorted_fgm['accuracy']
y_gm = _sorted_gm['accuracy']
y_centr = [99.59, 99.59, 99.59, 99.59, 99.59, 99.59]
plt.figure(figsize=(16, 12))
plt.plot(x, y_fgm, label="FGM", marker='D', linewidth=4)
plt.plot(x, y_gm, label="GM", marker='D', linewidth=4)
plt.plot(x, y_centr, label="Centralized", linewidth=4)
plt.xlabel('Workers', fontsize=FONT_SIZE)
plt.ylabel('Accuracy', fontsize=FONT_SIZE)
plt.legend(loc='best', fontsize=FONT_SIZE)
plt.grid(True)
plt.title('SFCC Dataset')
plt.xticks(x, fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.savefig('../results/sfc-plots/exp_Fig_3_1')

# Rounds
y_fgm = _sorted_fgm['rounds']
y_gm = _sorted_gm['rounds']
plt.figure(figsize=(16, 12))
plt.plot(x, y_fgm, label="FGM", marker='D', linewidth=4)
plt.plot(x, y_gm, label="GM", marker='D', linewidth=4)
plt.xlabel('Workers', fontsize=FONT_SIZE)
plt.ylabel('Rounds', fontsize=FONT_SIZE)
plt.legend(loc='best', fontsize=FONT_SIZE)
plt.grid(True)
plt.title('SFCC Dataset')
plt.xticks(x, fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.savefig('../results/sfc-plots/exp_Fig_3_2')

# Traffic
y_fgm = _sorted_fgm['traffic']
y_gm = _sorted_gm['traffic']
plt.figure(figsize=(16, 12))
plt.plot(x, y_fgm, label="FGM", marker='D', linewidth=4)
plt.plot(x, y_gm, label="GM", marker='D', linewidth=4)
plt.xlabel('Workers', fontsize=FONT_SIZE)
plt.ylabel('Traffic', fontsize=FONT_SIZE)
plt.legend(loc='best', fontsize=FONT_SIZE)
plt.grid(True)
plt.title('SFCC Dataset')
plt.xticks(x, fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.savefig('../results/sfc-plots/exp_Fig_3_3')

# Amazon Fine Food Reviews Dataset
sf1_fgm_df = pd.read_csv('../results/SF1_fgm-amazon.csv', usecols=fgm_headers)
sf2_fgm_df = pd.read_csv('../results/SF2_fgm-amazon.csv', usecols=fgm_headers)
sf1_gm_df = pd.read_csv('../results/SF1_gm-amazon.csv', usecols=gm_headers)
sf2_gm_df = pd.read_csv('../results/SF2_gm-amazon.csv', usecols=gm_headers)

# Model accuracy and rounds for various thresholds
# - Threshold: { 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1 }
# - Batch size: 16
# - Workers: 8
fgm_subdf = sf2_fgm_df[(sf2_fgm_df['id'] >= 1) & (sf2_fgm_df['id'] <= 7)]
_sorted_fgm = fgm_subdf.sort_values('threshold')
gm_subdf = sf2_gm_df[(sf2_gm_df['id'] >= 1) & (sf2_gm_df['id'] <= 7)]
_sorted_gm = gm_subdf.sort_values('threshold')

x = _sorted_fgm['threshold']

# Accuracy
y_fgm = _sorted_fgm['accuracy']
y_gm = _sorted_gm['accuracy']
y_centr = [98.37, 98.37, 98.37, 98.37, 98.37, 98.37, 98.37]
plt.figure(figsize=(16, 12))
plt.plot(x, y_fgm, label="FGM", marker='D', linewidth=4)
plt.plot(x, y_gm, label="GM", marker='D', linewidth=4)
plt.plot(x, y_centr, label="Centralized", linewidth=4)
plt.xlabel('Threshold', fontsize=FONT_SIZE)
plt.ylabel('Accuracy', fontsize=FONT_SIZE)
plt.legend(loc='best', fontsize=FONT_SIZE)
plt.grid(True)
plt.title('AFFR Dataset')
plt.xticks(x, fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.savefig('../results/amazon-plots/exp_Fig_1_1')

# Rounds
y_fgm = _sorted_fgm['rounds']
y_gm = _sorted_gm['rounds']
plt.figure(figsize=(16, 12))
plt.plot(x, y_fgm, label="FGM", marker='D', linewidth=4)
plt.plot(x, y_gm, label="GM", marker='D', linewidth=4)
plt.xlabel('Threshold', fontsize=FONT_SIZE)
plt.ylabel('Rounds', fontsize=FONT_SIZE)
plt.legend(loc='best', fontsize=FONT_SIZE)
plt.grid(True)
plt.title('AFFR Dataset')
plt.xticks(x, fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.savefig('../results/amazon-plots/exp_Fig_1_2')

# Traffic
y_fgm = _sorted_fgm['traffic']
y_gm = _sorted_gm['traffic']
plt.figure(figsize=(16, 12))
plt.plot(x, y_fgm, label="FGM", marker='D', linewidth=4)
plt.plot(x, y_gm, label="GM", marker='D', linewidth=4)
plt.xlabel('Threshold', fontsize=FONT_SIZE)
plt.ylabel('Traffic', fontsize=FONT_SIZE)
plt.legend(loc='best', fontsize=FONT_SIZE)
plt.grid(True)
plt.title('AFFR Dataset')
plt.xticks(x, fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.savefig('../results/amazon-plots/exp_Fig_1_3')

# Model accuracy and rounds for various batch sizes
# - Threshold: 0.5
# - Batch size: {1, 4, 16, 32, 64, 128}
# - Workers: 8
fgm_subdf = sf2_fgm_df[(sf2_fgm_df['id'] >= 8) & (sf2_fgm_df['id'] <= 13)]
_sorted_fgm = fgm_subdf.sort_values('batch-size')
gm_subdf = sf2_gm_df[(sf2_gm_df['id'] >= 8) & (sf2_gm_df['id'] <= 13)]
_sorted_gm = gm_subdf.sort_values('batch-size')

x = _sorted_fgm['batch-size']

# Accuracy
y_fgm = _sorted_fgm['accuracy']
y_gm = _sorted_gm['accuracy']
y_centr = [98.37, 98.37, 98.37, 98.37, 98.37, 98.37]
plt.figure(figsize=(16, 12))
plt.plot(x, y_fgm, label="FGM", marker='D', linewidth=4)
plt.plot(x, y_gm, label="GM", marker='D', linewidth=4)
plt.plot(x, y_centr, label="Centralized", linewidth=4)
plt.xlabel('Batch Size', fontsize=FONT_SIZE)
plt.ylabel('Accuracy', fontsize=FONT_SIZE)
plt.legend(loc='best', fontsize=FONT_SIZE)
plt.grid(True)
plt.title('AFFR Dataset')
plt.xticks(x, fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.savefig('../results/amazon-plots/exp_Fig_2_1')

# Rounds
y_fgm = _sorted_fgm['rounds']
y_gm = _sorted_gm['rounds']
plt.figure(figsize=(16, 12))
plt.plot(x, y_fgm, label="FGM", marker='D', linewidth=4)
plt.plot(x, y_gm, label="GM", marker='D', linewidth=4)
plt.xlabel('Batch Size', fontsize=FONT_SIZE)
plt.ylabel('Rounds', fontsize=FONT_SIZE)
plt.legend(loc='best', fontsize=FONT_SIZE)
plt.grid(True)
plt.title('AFFR Dataset')
plt.xticks(x, fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.savefig('../results/amazon-plots/exp_Fig_2_2')

# Traffic
y_fgm = _sorted_fgm['traffic']
y_gm = _sorted_gm['traffic']
plt.figure(figsize=(16, 12))
plt.plot(x, y_fgm, label="FGM", marker='D', linewidth=4)
plt.plot(x, y_gm, label="GM", marker='D', linewidth=4)
plt.xlabel('Batch Size', fontsize=FONT_SIZE)
plt.ylabel('Traffic', fontsize=FONT_SIZE)
plt.legend(loc='best', fontsize=FONT_SIZE)
plt.grid(True)
plt.title('AFFR Dataset')
plt.xticks(x, fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.savefig('../results/amazon-plots/exp_Fig_2_3')

# Model accuracy and rounds for various sites
# - Threshold: 0.5
# - Batch size: 16
# - Workers: {4, 8, 16, 32, 64, 128}
fgm_subdf = sf2_fgm_df[(sf2_fgm_df['id'] >= 14) & (sf2_fgm_df['id'] <= 19)]
_sorted_fgm = fgm_subdf.sort_values('sites')
gm_subdf = sf2_gm_df[(sf2_gm_df['id'] >= 14) & (sf2_gm_df['id'] <= 19)]
_sorted_gm = gm_subdf.sort_values('sites')

x = _sorted_fgm['sites']

# Accuracy
y_fgm = _sorted_fgm['accuracy']
y_gm = _sorted_gm['accuracy']
y_centr = [98.37, 98.37, 98.37, 98.37, 98.37, 98.37]
plt.figure(figsize=(16, 12))
plt.plot(x, y_fgm, label="FGM", marker='D', linewidth=4)
plt.plot(x, y_gm, label="GM", marker='D', linewidth=4)
plt.plot(x, y_centr, label="Centralized", linewidth=4)
plt.xlabel('Workers', fontsize=FONT_SIZE)
plt.ylabel('Accuracy', fontsize=FONT_SIZE)
plt.legend(loc='best', fontsize=FONT_SIZE)
plt.grid(True)
plt.title('AFFR Dataset')
plt.xticks(x, fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.savefig('../results/amazon-plots/exp_Fig_3_1')

# Rounds
y_fgm = _sorted_fgm['rounds']
y_gm = _sorted_gm['rounds']
plt.figure(figsize=(16, 12))
plt.plot(x, y_fgm, label="FGM", marker='D', linewidth=4)
plt.plot(x, y_gm, label="GM", marker='D', linewidth=4)
plt.xlabel('Workers', fontsize=FONT_SIZE)
plt.ylabel('Rounds', fontsize=FONT_SIZE)
plt.legend(loc='best', fontsize=FONT_SIZE)
plt.grid(True)
plt.title('AFFR Dataset')
plt.xticks(x, fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.savefig('../results/amazon-plots/exp_Fig_3_2')

# Traffic
y_fgm = _sorted_fgm['traffic']
y_gm = _sorted_gm['traffic']
plt.figure(figsize=(16, 12))
plt.plot(x, y_fgm, label="FGM", marker='D', linewidth=4)
plt.plot(x, y_gm, label="GM", marker='D', linewidth=4)
plt.xlabel('Workers', fontsize=FONT_SIZE)
plt.ylabel('Traffic', fontsize=FONT_SIZE)
plt.legend(loc='best', fontsize=FONT_SIZE)
plt.grid(True)
plt.title('AFFR Dataset')
plt.xticks(x, fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.savefig('../results/amazon-plots/exp_Fig_3_3')

# Comparison between safe functions over the same protocol
fgm_headers = ['id', 'threshold', 'batch-size', 'sites', 'rounds', 'subrounds', 'accuracy', 'traffic']

sf1_fgm_df = pd.read_csv('../results/SF1_fgm-sfc.csv', usecols=fgm_headers)
sf2_fgm_df = pd.read_csv('../results/SF2_fgm-sfc.csv', usecols=fgm_headers)

# Model accuracy and rounds for various thresholds
# - Threshold: { 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1 }
# - Batch size: 16
# - Workers: 8
sf1_fgm_subdf = sf1_fgm_df[(sf1_fgm_df['id'] >= 1) & (sf1_fgm_df['id'] <= 7)]
_sorted_fgm = sf1_fgm_subdf.sort_values('threshold')
sf2_fgm_subdf = sf2_fgm_df[(sf2_fgm_df['id'] >= 1) & (sf2_fgm_df['id'] <= 7)]
_sorted_gm = sf2_fgm_subdf.sort_values('threshold')

x = _sorted_fgm['threshold']

# Accuracy
y_sf1_fgm = _sorted_fgm['accuracy']
y_sf2_fgm = _sorted_gm['accuracy']
y_centr = [99.59, 99.59, 99.59, 99.59, 99.59, 99.59, 99.59]
plt.figure(figsize=(16, 12))
plt.plot(x, y_sf1_fgm, label="FGM_SF1", marker='D', linewidth=4)
plt.plot(x, y_sf2_fgm, label="FGM_SF2", marker='D', linewidth=4)
plt.plot(x, y_centr, label="Centralized", linewidth=4)
plt.xlabel('Threshold', fontsize=FONT_SIZE)
plt.ylabel('Accuracy', fontsize=FONT_SIZE)
plt.legend(loc='best', fontsize=FONT_SIZE)
plt.grid(True)
plt.xticks(x, fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.savefig('../results/sf-comp/exp_Fig_1_1')

# Rounds
y_sf1_fgm = _sorted_fgm['rounds']
y_sf2_fgm = _sorted_gm['rounds']
plt.figure(figsize=(16, 12))
plt.plot(x, y_sf1_fgm, label="FGM_SF1", marker='D', linewidth=4)
plt.plot(x, y_sf2_fgm, label="FGM_SF2", marker='D', linewidth=4)
plt.xlabel('Threshold', fontsize=FONT_SIZE)
plt.ylabel('Rounds', fontsize=FONT_SIZE)
plt.legend(loc='best', fontsize=FONT_SIZE)
plt.grid(True)
plt.xticks(x, fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.savefig('../results/sf-comp/exp_Fig_1_2')

# Traffic
y_sf1_fgm = _sorted_fgm['traffic']
y_sf2_fgm = _sorted_gm['traffic']
plt.figure(figsize=(16, 12))
plt.plot(x, y_sf1_fgm, label="FGM_SF1", marker='D', linewidth=4)
plt.plot(x, y_sf2_fgm, label="FGM_SF2", marker='D', linewidth=4)
plt.xlabel('Threshold', fontsize=FONT_SIZE)
plt.ylabel('Traffic', fontsize=FONT_SIZE)
plt.legend(loc='best', fontsize=FONT_SIZE)
plt.grid(True)
plt.xticks(x, fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.savefig('../results/sf-comp/exp_Fig_1_3')

# Model accuracy and rounds for various batch sizes
# - Threshold: 0.5
# - Batch size: {1, 4, 16, 32, 64, 128}
# - Workers: 8
sf1_fgm_subdf = sf1_fgm_df[(sf1_fgm_df['id'] >= 8) & (sf1_fgm_df['id'] <= 13)]
_sorted_fgm = sf1_fgm_subdf.sort_values('batch-size')
sf2_fgm_subdf = sf2_fgm_df[(sf2_fgm_df['id'] >= 8) & (sf2_fgm_df['id'] <= 13)]
_sorted_gm = sf2_fgm_subdf.sort_values('batch-size')

x = _sorted_fgm['batch-size']

# Accuracy
y_sf1_fgm = _sorted_fgm['accuracy']
y_sf2_fgm = _sorted_gm['accuracy']
y_centr = [99.59, 99.59, 99.59, 99.59, 99.59, 99.59]
plt.figure(figsize=(16, 12))
plt.plot(x, y_sf1_fgm, label="FGM_SF1", marker='D', linewidth=4)
plt.plot(x, y_sf2_fgm, label="FGM_SF2", marker='D', linewidth=4)
plt.plot(x, y_centr, label="Centralized", linewidth=4)
plt.xlabel('Batch Size', fontsize=FONT_SIZE)
plt.ylabel('Accuracy', fontsize=FONT_SIZE)
plt.legend(loc='best', fontsize=FONT_SIZE)
plt.grid(True)
plt.xticks(x, fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.savefig('../results/sf-comp/exp_Fig_2_1')

# Rounds
y_sf1_fgm = _sorted_fgm['rounds']
y_sf2_fgm = _sorted_gm['rounds']
plt.figure(figsize=(16, 12))
plt.plot(x, y_sf1_fgm, label="FGM_SF1", marker='D', linewidth=4)
plt.plot(x, y_sf2_fgm, label="FGM_SF2", marker='D', linewidth=4)
plt.xlabel('Batch Size', fontsize=FONT_SIZE)
plt.ylabel('Rounds', fontsize=FONT_SIZE)
plt.legend(loc='best', fontsize=FONT_SIZE)
plt.grid(True)
plt.xticks(x, fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.savefig('../results/sf-comp/exp_Fig_2_2')

# Traffic
y_sf1_fgm = _sorted_fgm['traffic']
y_sf2_fgm = _sorted_gm['traffic']
plt.figure(figsize=(16, 12))
plt.plot(x, y_sf1_fgm, label="FGM_SF1", marker='D', linewidth=4)
plt.plot(x, y_sf2_fgm, label="FGM_SF2", marker='D', linewidth=4)
plt.xlabel('Batch Size', fontsize=FONT_SIZE)
plt.ylabel('Traffic', fontsize=FONT_SIZE)
plt.legend(loc='best', fontsize=FONT_SIZE)
plt.grid(True)
plt.xticks(x, fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.savefig('../results/sf-comp/exp_Fig_2_3')

# Model accuracy and rounds for various sites
# - Threshold: 0.5
# - Batch size: 16
# - Workers: {4, 8, 16, 32, 64, 128}
sf1_fgm_subdf = sf1_fgm_df[(sf1_fgm_df['id'] >= 14) & (sf1_fgm_df['id'] <= 19)]
_sorted_fgm = sf1_fgm_subdf.sort_values('sites')
sf2_fgm_subdf = sf2_fgm_df[(sf2_fgm_df['id'] >= 14) & (sf2_fgm_df['id'] <= 19)]
_sorted_gm = sf2_fgm_subdf.sort_values('sites')

x = _sorted_fgm['sites']

# Accuracy
y_sf1_fgm = _sorted_fgm['accuracy']
y_sf2_fgm = _sorted_gm['accuracy']
y_centr = [99.59, 99.59, 99.59, 99.59, 99.59, 99.59]
plt.figure(figsize=(16, 12))
plt.plot(x, y_sf1_fgm, label="FGM_SF1", marker='D', linewidth=4)
plt.plot(x, y_sf2_fgm, label="FGM_SF2", marker='D', linewidth=4)
plt.plot(x, y_centr, label="Centralized", linewidth=4)
plt.xlabel('Workers', fontsize=FONT_SIZE)
plt.ylabel('Accuracy', fontsize=FONT_SIZE)
plt.legend(loc='best', fontsize=FONT_SIZE)
plt.grid(True)
plt.xticks(x, fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.savefig('../results/sf-comp/exp_Fig_3_1')

# Rounds
y_sf1_fgm = _sorted_fgm['rounds']
y_sf2_fgm = _sorted_gm['rounds']
plt.figure(figsize=(16, 12))
plt.plot(x, y_sf1_fgm, label="FGM_SF1", marker='D', linewidth=4)
plt.plot(x, y_sf2_fgm, label="FGM_SF2", marker='D', linewidth=4)
plt.xlabel('Workers', fontsize=FONT_SIZE)
plt.ylabel('Rounds', fontsize=FONT_SIZE)
plt.legend(loc='best', fontsize=FONT_SIZE)
plt.grid(True)
plt.xticks(x, fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.savefig('../results/sf-comp/exp_Fig_3_2')

# Traffic
y_sf1_fgm = _sorted_fgm['traffic']
y_sf2_fgm = _sorted_gm['traffic']
plt.figure(figsize=(16, 12))
plt.plot(x, y_sf1_fgm, label="FGM_SF1", marker='D', linewidth=4)
plt.plot(x, y_sf2_fgm, label="FGM_SF2", marker='D', linewidth=4)
plt.xlabel('Workers', fontsize=FONT_SIZE)
plt.ylabel('Traffic', fontsize=FONT_SIZE)
plt.legend(loc='best', fontsize=FONT_SIZE)
plt.grid(True)
plt.xticks(x, fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.savefig('../results/sf-comp/exp_Fig_3_3')
