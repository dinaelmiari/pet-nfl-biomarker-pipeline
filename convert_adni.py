import subprocess
from pathlib import Path

raw_dicom_dir = Path("/home/delmiari/scratch/project/data/ADNI/")
output_nifti_dir = Path("/home/delmiari/scratch/project/data/nifti_t1/")
output_nifti_dir.mkdir(parents=True, exist_ok=True)

target_subjects = {
    "002_S_0685", "002_S_0729", "002_S_1155", "003_S_1122", "005_S_0546", "005_S_0553", "005_S_0602", "005_S_0610",
    "006_S_0498", "006_S_1130", "010_S_0419", "010_S_0420", "014_S_0519", "014_S_0520", "014_S_0563", "014_S_0658",
    "016_S_0359", "016_S_0702", "021_S_0159", "021_S_0337", "021_S_0626", "021_S_0984", "022_S_0096", "022_S_0130",
    "023_S_0031", "023_S_0061", "023_S_0217", "023_S_0376", "023_S_0887", "023_S_0926", "023_S_1046", "024_S_1063",
    "027_S_0074", "027_S_0120", "029_S_0845", "029_S_1318", "031_S_0618", "031_S_0830", "031_S_0867", "032_S_0214",
    "032_S_1169", "035_S_0997", "036_S_0672", "036_S_0673", "036_S_0813", "036_S_0945", "037_S_0377", "037_S_1078",
    "041_S_1010", "052_S_0671", "052_S_1346", "052_S_1352", "053_S_0919", "057_S_0934", "057_S_1007", "057_S_1269",
    "073_S_0089", "073_S_0746", "094_S_1417", "098_S_0667", "100_S_0069", "100_S_0296", "100_S_1226", "100_S_1286",
    "114_S_0166", "116_S_0657", "116_S_0834", "116_S_1232", "123_S_1300", "127_S_0112", "127_S_0259", "127_S_0260",
    "127_S_0925", "127_S_1419", "128_S_0200", "128_S_0225", "128_S_0227", "128_S_0863", "128_S_1408", "130_S_0289",
    "131_S_0123", "137_S_0722", "137_S_0800", "137_S_0972", "137_S_0973", "137_S_0994", "137_S_1414"
}

converted_count = 0
missing = []

for sub_id in target_subjects:
    sub_path = raw_dicom_dir / sub_id
    if sub_path.exists():
        sub_out_dir = output_nifti_dir / sub_id
        sub_out_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = ["dcm2niix", "-z", "y", "-f", "%p_%s", "-o", str(sub_out_dir), str(sub_path)]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        converted_count += 1
        print(f"[{converted_count}/{len(target_subjects)}] Converted: {sub_id}")
    else:
        missing.append(sub_id)

print(f"\nFinished converting {converted_count} subjects.")
if missing:
    print(f"Missing from DICOM folder ({len(missing)}): {missing}")
