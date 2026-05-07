import os
import numpy as np
import wfdb
from scipy.spatial.distance import cosine
from app.core.signal import Signal

def generate_validation_csv(output_file='validation_subtypes.csv'):
    with open(output_file, 'w') as out:
        out.write("record,sample,true_subtype\n")
        
        db_path = os.path.join('DB', 'valdb')
        with open(os.path.join(db_path, 'RECORDS'), 'r') as f:
            records = [line.strip() for line in f if line.strip() and line.strip().startswith('II/')]
            
        last_pvc_vec = None
        
        for rec_path in records:
            full_path = os.path.join(db_path, rec_path)
            try:
                sig = Signal(record_path=full_path)
                r_peaks = [ann for ann in sig.annotations if ann['symbol'] == 'V']
                rr_intervals = np.diff([p['sample'] for p in sig.annotations]).tolist()
                rr_intervals.insert(0, int(0.8 * 360))
                
                median_rr = np.median(rr_intervals)
                
                for i, ann in enumerate(r_peaks):
                    peak = ann['sample']
                    rr_prev = rr_intervals[i]
                    rr_next = rr_intervals[i+1] if i+1 < len(rr_intervals) else median_rr
                    
                    # Правила подтипов
                    prev_rr_sec = rr_prev / 360.0
                    qt_est = 0.4 * np.sqrt(prev_rr_sec) * 360
                    
                    if rr_prev < (0.85 * qt_est): subtype = 'r'
                    elif (rr_prev + rr_next) < (2.0 * median_rr): subtype = 'i'
                    else:
                        seg = sig.get_segment(peak, 288)
                        if seg is not None and last_pvc_vec is not None:
                            sim = 1 - cosine(last_pvc_vec, seg)
                            subtype = 'M' if sim > 0.7 else 'P'
                        else: subtype = 'M'
                        
                    if subtype in ['M', 'P'] and sig.get_segment(peak, 288) is not None:
                        last_pvc_vec = sig.get_segment(peak, 288)
                        
                    out.write(f"{rec_path},{peak},{subtype}\n")
            except: pass

if __name__ == "__main__":
    generate_validation_csv()
    print("Сгенерирован файл validation_subtypes.csv")