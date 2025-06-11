#!/usr/bin/env python3
# Copyright (c) Opendatalab. All rights reserved.
import os
import time
import argparse
import sys
from pathlib import Path

from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset, ImageDataset
from magic_pdf.model.doc_analyze_by_custom_model_llm import doc_analyze_llm
from magic_pdf.model.custom_model import MonkeyOCR


def parse_pdf(input_file, output_dir, config_path):
    """
    Parse PDF file and save results
    
    Args:
        input_file: Input PDF file path
        output_dir: Output directory
        config_path: Configuration file path
    """
    print(f"Starting to parse file: {input_file}")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file does not exist: {input_file}")
    
    # Initialize model
    print("Loading model...")
    MonkeyOCR_model = MonkeyOCR(config_path)
    
    # Get filename
    name_without_suff = os.path.basename(input_file).split(".")[0]
    
    # Prepare output directory
    local_image_dir = os.path.join(output_dir, name_without_suff, "images")
    local_md_dir = os.path.join(output_dir, name_without_suff)
    image_dir = os.path.basename(local_image_dir)
    os.makedirs(local_image_dir, exist_ok=True)
    os.makedirs(local_md_dir, exist_ok=True)
    
    print(f"Output dir: {local_md_dir}")
    image_writer = FileBasedDataWriter(local_image_dir)
    md_writer = FileBasedDataWriter(local_md_dir)
    
    # Read file content
    reader = FileBasedDataReader()
    file_bytes = reader.read(input_file)
    
    # Create dataset instance
    file_extension = input_file.split(".")[-1].lower()
    if file_extension == "pdf":
        ds = PymuDocDataset(file_bytes)
    else:
        ds = ImageDataset(file_bytes)
    
    # Start inference
    print("Performing document parsing...")
    start_time = time.time()
    
    infer_result = ds.apply(doc_analyze_llm, MonkeyOCR_model=MonkeyOCR_model)
    
    # Pipeline processing
    pipe_result = infer_result.pipe_ocr_mode(image_writer, MonkeyOCR_model=MonkeyOCR_model)
    
    parsing_time = time.time() - start_time
    print(f"Parsing time: {parsing_time:.2f}s")

    infer_result.draw_model(os.path.join(local_md_dir, f"{name_without_suff}_model.pdf"))
    
    pipe_result.draw_layout(os.path.join(local_md_dir, f"{name_without_suff}_layout.pdf"))

    pipe_result.draw_span(os.path.join(local_md_dir, f"{name_without_suff}_spans.pdf"))

    pipe_result.dump_md(md_writer, f"{name_without_suff}.md", image_dir)
    
    pipe_result.dump_content_list(md_writer, f"{name_without_suff}_content_list.json", image_dir)

    pipe_result.dump_middle_json(md_writer, f'{name_without_suff}_middle.json')
    
    print("Results saved to ", local_md_dir)
    return local_md_dir


def batch_process_pdfs(input_dir, output_dir, config_path, recursive=False):
    """
    批量处理目录中的所有PDF文件
    
    Args:
        input_dir: 包含PDF文件的目录
        output_dir: 输出目录
        config_path: 配置文件路径
        recursive: 是否递归处理子目录中的文件
    
    Returns:
        处理成功的输出目录列表
    """
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"输入路径不是一个目录: {input_dir}")
    
    # 查找所有PDF文件
    pdf_files = []
    if recursive:
        # 递归搜索子目录中的PDF文件
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
    else:
        # 只搜索顶层目录中的PDF文件
        pdf_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                    if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(input_dir, f))]
    
    if not pdf_files:
        print(f"在 {input_dir} 中没有找到PDF文件")
        return []
    
    print(f"找到 {len(pdf_files)} 个PDF文件待处理")
    
    # 处理每个PDF文件
    results = []
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n处理文件 {i}/{len(pdf_files)}: {pdf_path}")
        
        try:
            result_dir = parse_pdf(pdf_path, output_dir, config_path)
            results.append(result_dir)
            print(f"✅ 成功处理: {pdf_path}")
        except Exception as e:
            print(f"❌ 处理失败 {pdf_path}: {str(e)}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="PDF文档解析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python parse.py input.pdf
  python parse.py input.pdf -o ./output
  python parse.py input.pdf -c model_configs.yaml
  python parse.py -b input_directory -o ./output
  python parse.py -b input_directory -r -o ./output
        """
    )
    
    parser.add_argument(
        "input_path",
        nargs="?",
        help="输入PDF文件路径或目录(与-b一起使用时)"
    )
    
    parser.add_argument(
        "-b", "--batch",
        action="store_true",
        help="批量处理输入目录中的所有PDF文件"
    )
    
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="递归处理子目录中的PDF文件(仅与-b一起使用)"
    )
    
    parser.add_argument(
        "-o", "--output",
        default="./output",
        help="输出目录 (默认: ./output)"
    )
    
    parser.add_argument(
        "-c", "--config",
        default="model_configs.yaml",
        help="配置文件路径 (默认: model_configs.yaml)"
    )
    
    args = parser.parse_args()
    
    if not args.input_path:
        parser.error("需要提供输入路径")
    
    if args.recursive and not args.batch:
        parser.error("-r/--recursive 选项只能与 -b/--batch 一起使用")
    
    try:
        if args.batch:
            # 批处理模式
            print(f"开始批量处理目录中的PDF文件: {args.input_path}")
            result_dirs = batch_process_pdfs(
                args.input_path,
                args.output,
                args.config,
                recursive=args.recursive
            )
            
            if result_dirs:
                print(f"\n✅ 批处理完成! 成功处理 {len(result_dirs)} 个文件。")
                print(f"结果保存在: {args.output}")
            else:
                print("\n⚠️ 没有成功处理任何文件。")
                
        else:
            # 单文件处理模式
            result_dir = parse_pdf(
                args.input_path,
                args.output,
                args.config
            )
            print(f"\n✅ 解析完成! 结果保存在: {result_dir}")
        
    except Exception as e:
        print(f"\n❌ 处理失败: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
