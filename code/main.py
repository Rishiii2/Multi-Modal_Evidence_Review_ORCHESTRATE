import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from data_loader import load_data, prepare_claim_context
from agent import process_claim

logging.basicConfig(level=logging.INFO, filename='../log.txt', format='%(asctime)s - %(levelname)s - %(message)s')

def main(csv_path="dataset/claims.csv", output_path="output.csv", max_workers=5):
    logging.info("Starting Multi-Modal Evidence Review Pipeline")
    
    # Load all datasets
    datasets = load_data(base_path="dataset")
    claims_df = pd.read_csv(csv_path)
    
    results = []
    
    # Prepare contexts
    contexts = []
    for _, row in claims_df.iterrows():
        ctx = prepare_claim_context(row, datasets["history"], datasets["evidence"])
        contexts.append(ctx)
        
    logging.info(f"Processing {len(contexts)} claims concurrently with {max_workers} workers...")
    
    # Process concurrently to reduce latency
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ctx = {executor.submit(process_claim, ctx): ctx for ctx in contexts}
        
        for future in tqdm(as_completed(future_to_ctx), total=len(contexts), desc="Processing Claims"):
            ctx = future_to_ctx[future]
            try:
                res = future.result()
                
                # Format output row
                risk_flags_str = ";".join(res.get("risk_flags", ["none"])) if res.get("risk_flags") else "none"
                supporting_images_str = ";".join(res.get("supporting_image_ids", ["none"])) if res.get("supporting_image_ids") else "none"
                
                output_row = {
                    "user_id": ctx["user_id"],
                    "image_paths": ";".join(ctx["image_paths"]),
                    "user_claim": ctx["user_claim"],
                    "claim_object": ctx["claim_object"],
                    "evidence_standard_met": res.get("evidence_standard_met", False),
                    "evidence_standard_met_reason": res.get("evidence_standard_met_reason", ""),
                    "risk_flags": risk_flags_str,
                    "issue_type": res.get("issue_type", "unknown"),
                    "object_part": res.get("object_part", "unknown"),
                    "claim_status": res.get("claim_status", "not_enough_information"),
                    "claim_status_justification": res.get("claim_status_justification", ""),
                    "supporting_image_ids": supporting_images_str,
                    "valid_image": res.get("valid_image", False),
                    "severity": res.get("severity", "unknown")
                }
                results.append(output_row)
            except Exception as e:
                logging.error(f"Failed to process claim for user {ctx['user_id']}: {e}")

    # Build DataFrame
    out_df = pd.DataFrame(results)
    
    # Ensure correct column order
    cols = [
        "user_id", "image_paths", "user_claim", "claim_object", 
        "evidence_standard_met", "evidence_standard_met_reason", 
        "risk_flags", "issue_type", "object_part", "claim_status", 
        "claim_status_justification", "supporting_image_ids", 
        "valid_image", "severity"
    ]
    
    # Add any missing columns just in case
    for col in cols:
        if col not in out_df.columns:
            out_df[col] = None
            
    out_df = out_df[cols]
    
    import csv
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in results:
            writer.writerow({col: row.get(col, "") for col in cols})
    
    logging.info(f"Output saved to {output_path}")
    print(f"Processing complete! Output saved to {output_path}")

if __name__ == "__main__":
    main()
