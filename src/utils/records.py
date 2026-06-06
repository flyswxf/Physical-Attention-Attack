import csv

def save_records_to_csv(records, output_path):
    """
    将结构化实验记录保存为 CSV。
    """
    fieldnames = list(records[0].keys())
    with open(output_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def load_records_from_csv(csv_path):
    """
    从 CSV 读取实验记录，并恢复基础类型。
    """
    records = []
    with open(csv_path, "r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            parsed_row = {}
            for key, value in row.items():
                if value == "":
                    raise ValueError(f"CSV 字段为空: key={key}")
                if key in NUMERIC_FIELDS:
                    parsed_row[key] = float(value)
                else:
                    parsed_row[key] = value
            records.append(parsed_row)
    return records