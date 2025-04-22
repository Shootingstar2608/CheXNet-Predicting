import os

# Đường dẫn đến thư mục chứa ảnh
image_dir = "ChestX-ray14/images/"

# Bước 1: Lấy danh sách tất cả file ảnh (.png) trong thư mục
image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]

# Lưu danh sách tên ảnh vào một file (để kiểm tra nếu cần)
with open("available_images.txt", "w") as f:
    for img in image_files:
        f.write(img + "\n")

print(f"Đã tìm thấy {len(image_files)} ảnh trong thư mục.")

# Chuyển danh sách tên ảnh thành một tập hợp (set) để tìm kiếm nhanh
available_images = set(image_files)

# Bước 2: Hàm để lọc file danh sách và tạo file mới
def filter_list_file(input_file, output_file):
    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            # Lấy tên ảnh từ cột đầu tiên
            image_name = line.split()[0]
            # Nếu ảnh tồn tại trong thư mục, giữ lại dòng đó
            if image_name in available_images:
                f_out.write(line)

# Đường dẫn đến các file gốc và file mới
list_files = [
    ("ChestX-ray14/labels/test_list.txt", "ChestX-ray14/labels/test_list_filtered.txt"),
    ("ChestX-ray14/labels/train_list.txt", "ChestX-ray14/labels/train_list_filtered.txt"),
    ("ChestX-ray14/labels/val_list.txt", "ChestX-ray14/labels/val_list_filtered.txt")
]

# Lọc từng file
for input_file, output_file in list_files:
    filter_list_file(input_file, output_file)
    print(f"Đã tạo file mới: {output_file}")