

input_path = './data/新企业专利总数.xlsx'
output_path = './data/新企业专利总数_去重.xlsx'
CACHE_FILE = "qcc_cache.json"  # 新增缓存文件
QICHACHA_API_KEY = "YOUR_API_KEY"  # 替换为你的API密钥
EXCLUDED_REGIONS = ['香港', '澳门', '台湾', '海外', '境外']

if __name__ == "__main__":
    print("=====================================")
