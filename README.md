# 注意事项：
1.使用用户点赞动态的行为数据产生用户的特征向量<br>
2.类数据库中物品的CLUB标签与用户加入的CLUB标签为二级标签，精选集的CLUB标签为二级标签，存储的物品只为动态内容<br>
3.用户行为和物品信息由src/data_base/check/behavior.xlsx表格维护<br>
4.buffer路径中存储item_cf、user_cf和hybrid_cf模型的中间运行结果：
item_cf_top_n_recommendation_map.feather存储item_cf模型的中间运行结果；
user_cf_top_n_recommendation_map.feather存储user_cf模型的中间运行结果；
item_cf_in_hybrid_cf_top_n_recommendation_map.feather和user_cf_top_n_recommendation_map.feather存储hybrid_cf模型的中间运行结果<br>
5.algo_results/perf_result路径中存储推荐性能指标结果<br>
6.algo_results/rec_result路径中存储推荐物品的结果<br>
7.algo_results/tuning_result路径中存储模型调参的结果<br>
8.算法名称中"generalized_cf"对应混合广义协同过滤；
"item_cf"对应基于物品的协同过滤；
"user_cf"对应基于用户的协同过滤；
"hybrid_cf"对应混合协同过滤

# 运行代码的命令：
1.运行主程序：python main.py<br>
2.运行GUI程序：python aifunwatcher.py

# 类数据库实例如下
```json
data = {
    "user": {
        "1001": {
            ("follow", "user"): ["1002", "1003"],
            ("followed", "user"): ["1002"], 
            ("view", "item"): ["动态:0001", "动态:0002"],
            ("comment", "item"): ["动态:0002"],
            ("join", "club"): ["动态:二级标签:剧本杀:剧本杀"]，
        },
        "1002": {
            ("follow", "user"): ["1001"],
            ("followed", "user"): ["1001"], 
        },
        "1003": {
            ("followed", "user"): ["1001"], 
        }
    }, 
    "item": {
        "动态:0001": {
            ("viewed", "user"): ["1001"],
            ("have", "club"): ["动态:二级标签:其他:王者"],
        } , 
        "动态:0002": {
            ("viewed", "user"): ["1001"], 
            ("commented", "user"): ["1001"],
            ("have", "club"): ["动态:二级标签:剧本杀:独家", "动态:二级标签:剧本杀:剧本杀"],
            ("have", "publish_time"): "2021-09-30 18:49:30",
        } , 
        "动态:0003": {
            ("have", "club"): ["动态:二级标签:剧本杀:剧本杀"],
        }  
    },
    "club": {
        "动态:二级标签:剧本杀": {
            ("include", "item"): ["动态:0002", "动态:0003"]
            ("have", "selected_item"): ["动态:0002"]
        },
        "动态:二级标签:其他": {
            ("include", "item"): ["动态:0001"]
        },
        "动态:推广": {
            ('have', 'selected_item'): ['动态:0002', '动态:0005', '动态:0009']
        }
    }
}
