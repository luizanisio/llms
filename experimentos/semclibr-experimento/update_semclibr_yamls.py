import os
import glob

yaml_files = glob.glob("/mnt/d/wsl_dev/llms/experimentos/semclibr-experimento/03_compara_*.yaml") + \
             glob.glob("/mnt/d/wsl_dev/llms/experimentos/semclibr-experimento/06_compara_*.yaml")

for file in yaml_files:
    with open(file, "r") as f:
        lines = f.readlines()
        
    new_lines = []
    skip_mode = False
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Stop skip mode if we reach a new metric under campos:
        if skip_mode:
            if line.strip() and not line.strip().startswith("-") and line.startswith("    "):
                skip_mode = False
            else:
                i += 1
                continue

        if line.strip().startswith("bertscore:") or line.strip().startswith("xbertscore:"):
            skip_mode = True
            i += 1
            continue
            
        if line.strip().startswith("sbert_grande:") or line.strip().startswith("sentence_bert:"):
            skip_mode = True
            i += 1
            continue
            
        # Remove individual lines
        if "bertscore_batch_size:" in line:
            i += 1
            continue
        if "sbert_batch_size:" in line:
            i += 1
            continue
        if "#bertscore: \"pucpr/biobertpt-all\"" in line:
            i += 1
            continue
        if line.strip() == "- bertscore" or line.strip() == "- sbert_grande" or line.strip() == "- sentence_bert":
            i += 1
            continue
            
        # Also clean up the two comment lines if they exist, but maybe too hard. Let's just keep comments or remove them if they refer to bertscore.
        if "Alternativa para pt-br clínico" in line or "senão a dificuldade e a avaliação final ficam" in line:
            # Look ahead to see if #bertscore follows
            if i + 2 < len(lines) and "#bertscore:" in lines[i+2]:
                i += 3
                continue
            if i + 1 < len(lines) and "#bertscore:" in lines[i+1]:
                i += 2
                continue
                
        new_lines.append(line)
        i += 1
        
    with open(file, "w") as f:
        f.writelines(new_lines)
    print(f"Processed {os.path.basename(file)}")

