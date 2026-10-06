import argparse

from pymilvus import utility

from . import milvus_store


def main() -> None:
    parser = argparse.ArgumentParser(description="Tạo (hoặc kiểm tra) 1 collection Milvus.")
    parser.add_argument(
        "--name", "-n",
        default=milvus_store.COLLECTION_NAME,
        help=f"Tên collection cần tạo (mặc định lấy từ MILVUS_COLLECTION trong .env: {milvus_store.COLLECTION_NAME})",
    )
    args = parser.parse_args()

    milvus_store._connect()
    existed = utility.has_collection(args.name)
    collection = milvus_store.ensure_collection(args.name)

    if existed:
        print(f"Collection '{args.name}' đã tồn tại — không tạo lại.")
    else:
        print(f"Đã tạo collection '{args.name}' (dim={milvus_store.EMBEDDING_DIM}).")

    print(f"Milvus: {milvus_store.MILVUS_HOST}:{milvus_store.MILVUS_PORT}")
    print(f"Số record hiện có: {collection.num_entities}")

    if args.name != milvus_store.COLLECTION_NAME:
        print(
            f"\nLưu ý: pipeline OCR/chatbot hiện đang dùng collection '{milvus_store.COLLECTION_NAME}' "
            f"(biến MILVUS_COLLECTION trong .env). Muốn app dùng collection '{args.name}' vừa tạo, "
            f"cần đổi MILVUS_COLLECTION trong .env sang '{args.name}' rồi khởi động lại worker."
        )


if __name__ == "__main__":
    main()
