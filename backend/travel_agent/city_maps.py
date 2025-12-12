"""
全球城市映射表
中文 / 拼音 / 英文 / 机场码 → IATA 机场码 & 城市码
"""

CITY_NAME_TO_MAIN_AIRPORT = {
    # 国内
    "北京": "PEK", "北京市": "PEK", "beijing": "PEK",
    "上海": "PVG", "上海市": "PVG", "shanghai": "PVG",
    "广州": "CAN", "广州市": "CAN", "guangzhou": "CAN",
    "深圳": "SZX", "深圳市": "SZX", "shenzhen": "SZX",
    "成都": "CTU", "成都市": "CTU", "chengdu": "CTU",
    "杭州": "HGH", "杭州市": "HGH", "hangzhou": "HGH",
    "南京": "NKG", "南京市": "NKG", "nanjing": "NKG",
    "西安": "XIY", "西安市": "XIY", "xian": "XIY", "xi'an": "XIY",
    "重庆": "CKG", "重庆市": "CKG", "chongqing": "CKG",
    "青岛": "TAO", "青岛市": "TAO", "qingdao": "TAO",
    "厦门": "XMN", "厦门市": "XMN", "xiamen": "XMN",
    "天津": "TSN", "天津市": "TSN", "tianjin": "TSN",
    "大连": "DLC", "大连市": "DLC", "dalian": "DLC",

    # 亚洲
    "东京": "NRT", "東京": "NRT", "tokyo": "NRT",
    "大阪": "KIX", "osaka": "KIX",
    "首尔": "ICN", "首爾": "ICN", "seoul": "ICN",
    "釜山": "PUS", "busan": "PUS",
    "新加坡": "SIN", "singapore": "SIN",
    "曼谷": "BKK", "bangkok": "BKK",
    "吉隆坡": "KUL", "kuala lumpur": "KUL",
    "雅加达": "CGK", "jakarta": "CGK",
    "马尼拉": "MNL", "manila": "MNL",
    "胡志明": "SGN", "ho chi minh": "SGN",
    "迪拜": "DXB", "dubai": "DXB",
    "多哈": "DOH", "doha": "DOH",
    "伊斯坦布尔": "IST", "istanbul": "IST",
    "香港": "HKG", "香港特别行政区": "HKG",
    "hong kong": "HKG", "hongkong": "HKG", "xianggang": "HKG",


    # 欧洲
    "巴黎": "CDG", "paris": "CDG",
    "伦敦": "LHR", "london": "LHR",
    "罗马": "FCO", "rome": "FCO",
    "米兰": "MXP", "milan": "MXP",
    "阿姆斯特丹": "AMS", "amsterdam": "AMS",
    "马德里": "MAD", "madrid": "MAD",
    "巴塞罗那": "BCN", "barcelona": "BCN",
    "慕尼黑": "MUC", "munich": "MUC",
    "法兰克福": "FRA", "frankfurt": "FRA",
    "苏黎世": "ZRH", "zurich": "ZRH",
    "维也纳": "VIE", "vienna": "VIE",
    "布拉格": "PRG", "prague": "PRG",
    "雅典": "ATH", "athens": "ATH",

    # 美洲
    "纽约": "JFK", "紐約": "JFK", "new york": "JFK",
    "洛杉矶": "LAX", "los angeles": "LAX",
    "旧金山": "SFO", "san francisco": "SFO",
    "芝加哥": "ORD", "chicago": "ORD",
    "迈阿密": "MIA", "miami": "MIA",
    "拉斯维加斯": "LAS", "las vegas": "LAS",
    "西雅图": "SEA", "seattle": "SEA",
    "温哥华": "YVR", "vancouver": "YVR",
    "多伦多": "YYZ", "toronto": "YYZ",
    "墨西哥城": "MEX", "mexico city": "MEX",
    "圣保罗": "GRU", "sao paulo": "GRU",
    "布宜诺斯艾利斯": "EZE", "buenos aires": "EZE",

    # 大洋洲
    "悉尼": "SYD", "sydney": "SYD",
    "墨尔本": "MEL", "melbourne": "MEL",
    "奥克兰": "AKL", "auckland": "AKL",
}

CITY_NAME_TO_CITY_CODE = {
    # 国内
    "北京": "BJS", "北京市": "BJS", "beijing": "BJS",
    "上海": "SHA", "上海市": "SHA", "shanghai": "SHA",
    "广州": "CAN", "广州市": "CAN", "guangzhou": "CAN",
    "深圳": "SZX", "深圳市": "SZX", "shenzhen": "SZX",
    "成都": "CTU", "成都市": "CTU", "chengdu": "CTU",
    "杭州": "HGH", "杭州市": "HGH", "hangzhou": "HGH",
    "南京": "NKG", "南京市": "NKG", "nanjing": "NKG",
    "西安": "XIY", "西安市": "XIY", "xian": "XIY", "xi'an": "XIY",
    "重庆": "CKG", "重庆市": "CKG", "chongqing": "CKG",
    "青岛": "TAO", "青岛市": "TAO", "qingdao": "TAO",
    "厦门": "XMN", "厦门市": "XMN", "xiamen": "XMN",
    "天津": "TSN", "天津市": "TSN", "tianjin": "TSN",
    "大连": "DLC", "大连市": "DLC", "dalian": "DLC",

    # 亚洲
    "东京": "TYO", "東京": "TYO", "tokyo": "TYO",
    "大阪": "OSA", "osaka": "OSA",
    "首尔": "SEL", "首爾": "SEL", "seoul": "SEL",
    "釜山": "PUS", "busan": "PUS",
    "香港": "HKG", "香港特别行政区": "HKG",
    "新加坡": "SIN", "singapore": "SIN",
    "曼谷": "BKK", "bangkok": "BKK",
    "吉隆坡": "KUL", "kuala lumpur": "KUL",
    "雅加达": "JKT", "jakarta": "JKT",
    "马尼拉": "MNL", "manila": "MNL",
    "胡志明": "SGN", "ho chi minh": "SGN",
    "迪拜": "DXB", "dubai": "DXB",
    "多哈": "DOH", "doha": "DOH",
    "伊斯坦布尔": "IST", "istanbul": "IST",

    # 欧洲
    "巴黎": "PAR", "paris": "PAR",
    "伦敦": "LON", "london": "LON",
    "罗马": "ROM", "rome": "ROM",
    "米兰": "MIL", "milan": "MIL",
    "阿姆斯特丹": "AMS", "amsterdam": "AMS",
    "马德里": "MAD", "madrid": "MAD",
    "巴塞罗那": "BCN", "barcelona": "BCN",
    "慕尼黑": "MUC", "munich": "MUC",
    "法兰克福": "FRA", "frankfurt": "FRA",
    "苏黎世": "ZUR", "zurich": "ZUR",
    "维也纳": "VIE", "vienna": "VIE",
    "布拉格": "PRG", "prague": "PRG",
    "雅典": "ATH", "athens": "ATH",

    # 美洲
    "纽约": "NYC", "new york": "NYC",
    "洛杉矶": "LAX", "los angeles": "LAX",
    "旧金山": "SFO", "san francisco": "SFO",
    "芝加哥": "CHI", "chicago": "CHI",
    "迈阿密": "MIA", "miami": "MIA",
    "拉斯维加斯": "LAS", "las vegas": "LAS",
    "西雅图": "SEA", "seattle": "SEA",
    "温哥华": "YVR", "vancouver": "YVR",
    "多伦多": "YTO", "toronto": "YTO",
    "墨西哥城": "MEX", "mexico city": "MEX",
    "圣保罗": "SAO", "sao paulo": "SAO",
    "布宜诺斯艾利斯": "BUE", "buenos aires": "BUE",

    # 大洋洲
    "悉尼": "SYD", "sydney": "SYD",
    "墨尔本": "MEL", "melbourne": "MEL",
    "奥克兰": "AKL", "auckland": "AKL",
}

AIRPORT_TO_CITY_CODE = {
    # 国内
    "PEK": "BJS", "PKX": "BJS",
    "PVG": "SHA", "SHA": "SHA",
    "CAN": "CAN",
    "SZX": "SZX",
    "CTU": "CTU",
    "HGH": "HGH",
    "NKG": "NKG",
    "XIY": "XIY",
    "CKG": "CKG",
    "TAO": "TAO",
    "XMN": "XMN",
    "TSN": "TSN",
    "DLC": "DLC",

    # 亚洲
    "NRT": "TYO", "HND": "TYO",
    "KIX": "OSA",
    "ICN": "SEL", "GMP": "SEL",
    "SIN": "SIN",
    "BKK": "BKK",
    "KUL": "KUL",
    "CGK": "JKT",
    "MNL": "MNL",
    "SGN": "SGN",
    "DXB": "DXB",
    "DOH": "DOH",
    "IST": "IST",
    "HKG": "HKG",

    # 欧洲
    "CDG": "PAR", "ORY": "PAR",
    "LHR": "LON", "LGW": "LON", "STN": "LON",
    "FCO": "ROM", "CIA": "ROM",
    "MXP": "MIL", "LIN": "MIL",
    "AMS": "AMS",
    "MAD": "MAD",
    "BCN": "BCN",
    "MUC": "MUC",
    "FRA": "FRA",
    "ZRH": "ZUR",
    "VIE": "VIE",
    "PRG": "PRG",
    "ATH": "ATH",

    # 美洲
    "JFK": "NYC", "LGA": "NYC", "EWR": "NYC",
    "LAX": "LAX",
    "SFO": "SFO",
    "ORD": "CHI",
    "MIA": "MIA",
    "LAS": "LAS",
    "SEA": "SEA",
    "YVR": "YVR",
    "YYZ": "YTO",
    "MEX": "MEX",
    "GRU": "SAO",
    "EZE": "BUE",

    # 大洋洲
    "SYD": "SYD",
    "MEL": "MEL",
    "AKL": "AKL",
}