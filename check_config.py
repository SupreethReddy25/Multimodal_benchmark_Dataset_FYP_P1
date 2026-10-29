"""Quick sanity check: load all regions and print computed window indices."""
import sys
sys.path.insert(0, ".")

from src.pipeline.config import load_region, list_regions

regions = list_regions()
print(f"Found {len(regions)} regions: {regions}\n")

for rid in regions:
    r = load_region(rid)
    print(f"{'='*60}")
    print(f"  {rid}: {r['name']}")
    print(f"  Extent : {r['lat_min']}-{r['lat_max']}N  {r['lon_min']}-{r['lon_max']}E")
    print(f"  Diviner: rows {r['diviner_row_start']}-{r['diviner_row_end']}  "
          f"cols {r['diviner_col_start']}-{r['diviner_col_end']}  "
          f"({r['diviner_height']}x{r['diviner_width']} px)")
    print(f"  LOLA   : rows {r['lola_row_start']}-{r['lola_row_end']}  "
          f"cols {r['lola_col_start']}-{r['lola_col_end']}  "
          f"({r['lola_height']}x{r['lola_width']} px)")
    print(f"  Out    : {r['intermediate_dir']}")

print("\nAll regions loaded successfully.")
