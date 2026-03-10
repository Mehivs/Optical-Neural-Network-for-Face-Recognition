import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

margin = 2
out_path = '../output/0120_2023_margin2_w0_N1000_whole_detector_8x8_2_metalayer/'
out_data = pd.read_csv(os.path.join(out_path, "test_set_FR_FA.csv"))
thresholds = out_data['thresholds'].values
FR_FA_fig = plt.figure()
plt.plot(thresholds, out_data['false_reject'].values, label='False reject')
plt.plot(thresholds, out_data['false_accept'].values, label='False accept')
plt.plot(thresholds, out_data['false_reject'].values + out_data['false_accept'].values, label='Total error rate', c = '#C00000', linestyle = '--')
plt.ylim(-0.02, 0.6)
plt.legend()
plt.title(
    f"Best threshold: {out_data['best_threshold'].iloc[0]:.2f}, rate: {out_data['lowest_rate'].iloc[0]:.2f}")
#plt.show()
plt.savefig(os.path.join(out_path, f"FR_FA_with_total_acc.png"))
plt.close()
