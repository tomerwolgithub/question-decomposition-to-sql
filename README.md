# Weakly Supervised Text-to-SQL Parsing through Question Decomposition
The official repository for the paper [***"Weakly Supervised Text-to-SQL Parsing through Question Decomposition"***](https://arxiv.org/abs/2112.06311) by Tomer Wolfson, Daniel Deutch and Jonathan Berant, accepted to the **Finings of NAACL 2022**.

This repository contains the code and data used in our paper:

1. Code for automatically synthesizing SQL queries from question decompositions + answers
2. Code for the models used in our paper mapping text-to-SQL and text-to-QDMR 

* **Structure:**
	* [**Setup**](#setup-)
	* [**Resources**](#download-resources-%EF%B8%8F)
	* [**Data Generation**](#data-generation-)
	* [**Models**](#models-%EF%B8%8F)
	* [**Experiments**](#experiments-%EF%B8%8F)
	* [**Citation**](#citation-)
	* [**License**](#license)


## Setup 🙌🏼

1. Create the virtual environment
```
conda create -n [ENV_NAME] python=3.8
conda activate [ENV_NAME]
```

2. Clone the repository
```
git clone https://github.com/tomerwolgithub/question-decomposition-to-sql
cd question-decomposition-to-sql
```

3. Install the relevant requirements 
```
pip install -r requirements.txt 
python -m spacy download en_core_web_lg
```

4. To train the QDMR parser model please setup a separate environment (due to different Hugginface versions):
```
conda create -n qdmr_parser_env python=3.8
conda activate qdmr_parser_env
pip install -r requirements_qdmr_parser.txt 
python -m spacy download en_core_web_lg
```

## Download Resources 🗝️

### 1. QDMR Parsing Datasets:
* Academic, GeoQuery and Spider gold QDMRs from the [Break dataset](https://allenai.github.io/Break)
* Gold QDMR data for IMDB and Yelp is available [in this repository](https://github.com/tomerwolgithub/question-decomposition-to-sql/tree/main/data/annotated_qdmr)

### 2. Text-to-SQL Datasets:
* [Spider](https://yale-lily.github.io/spider)
* [Academic, IMDB, Yelp and GeoQuery](https://github.com/jkkummerfeld/text2sql-data/tree/master/data/original) (Jonathan Kummerfeld's repository)

### 3. Databases (schema & contents):
* [Spider (Yu et al., 2018)](https://yale-lily.github.io/spider)
* [Academic, IMDB and Yelp (Yaghmazadeh et al., OOPSLA 2017)](https://drive.google.com/drive/folders/0B-2uoWxAwJGKY09kaEtTZU1nTWM)
* [GeoQuery (Zelle & Mooney, AAAI 1996)](https://github.com/jkkummerfeld/text2sql-data)

Convert the MySQL databases of Academic, IMDB, Yelp and GeoQuery to sqlite format using the tool of [Jean-Luc Lacroix](https://github.com/dumblob/mysql2sqlite):
```
./mysql2sqlite academic_mysql.sql | sqlite3 academic_sqlite.db
```

## Data Generation 🔨
Our SQL synthesis is given examples of `<QDMR, database, answer>` and automatically generates a SQL that executes to the correct answer.
The QDMR decompositions are either manually annotated or automatically predicted by a trained QDMR parser.

Begin by copying all relevant sqlite databases to the `data_generation` directory.
```bash
mkdir data_generation/data
mkdir data_generation/data/spider_databases # copy Spider databases here
mkdir data_generation/data/other_databases # copy Academic, IMDB, Yelp and Geo databases here
```

1. The SQL synthesis expects a formatted `csv` file, see [example](https://github.com/tomerwolgithub/question-decomposition-to-sql/blob/main/data/sql_synthesis_results/sql_synthesis_input_example.csv). Note that the SQL query in these files is only used to compute the answer.
2. This may take several hours, as multiple candidate SQL are being executed on their respective database.
3. To synthesize SQL from the `<QDMR, database, answer>` examples run:

```bash
python data_generation/main.py \
--input_file input_qdmr_examples.csv \
--output_file qdmr_grounded_sql.csv \
--json_steps True
```

### Synthesized Data
The SQL synthesized using QDMR + answer supervision is available for each dataset in the [`data/sql_synthesis_results/`](https://github.com/tomerwolgithub/question-decomposition-to-sql/tree/main/data/sql_synthesis_results) directory. 
* `data/sql_synthesis_results/gold_qdmr_supervision`: contains SQL synthesized using gold QDMRs that are manually annotated
* `data/sql_synthesis_results/predicted_qdmr_supervision`: contains SQL synthesized using QDMRs predicted by a trained parser



## Models 🗂️

### QDMR Parser
The QDMR parser is a T5-large sequence-to-sequence model that is finetuned to map questions to their QDMR. The model expects as input two `csv` files as its train and dev sets. Use the files from the downloaded Break dataset to train the parser. Make sure that you are in the relevant python environment (`requirements_qdmr_parser.txt`).

To train the QDMR parser configure the following parameters in `train.py`:
* `data_dir`: the path to the directory containing the NL to QDMR datasets
* `training_set_file`: name of the train set `csv` (e.g. `break_train.csv`)
* `dev_set_file`: name of the dev set `csv` (e.g. `break_dev.csv`)
* `output_dir`: the directory to store the trained model

After configuration, train the model as follows:
```bash
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 python src/qdmr_parser/train.py
```

To test a trained model and store its predictions, configure the following parameters in `test.py`:
* `checkpoint_path`: path to the trained QDMR parser model to be evaluated
* `dev_set_file`: name of the dev set `csv` to generate predictions for
* `predictions_output_file`: the output file to store the parser's generated predictions

And run the following command:
```bash
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 python src/qdmr_parser/test.py
```

### Text-to-SQL 
The text-to-SQL models are T5-large sequence-to-sequence models, finetuned to map questions to executable SQL queries.
We compare the models trained on gold SQL queries, annotated by experts, to our synthesized SQL from QDMR and answer supervision.

#### 1. Setup directory
Setup the data for the text-to-SQL experiments as follows:
```bash
data
├── tables.json			# Spider tables.json
└── databases
│   └── academic			
│       └── academic.sqlite	# Sqlite version of the populated Academic database (see downloads)
│   └── geo			
│       └── geo.sqlite		# Sqlite version of the populated Geo database (see downloads)
│   └── imdb			
│       └── imdb.sqlite		# Sqlite version of the populated IMDB database (see downloads)
│   └── spider_databases 	# Spider databases directory
│       └── activity_1
│           └── activity_1.sqlite
│       └── ...   
│   └── yelp			
│       └── yelp.sqlite		# Sqlite version of the populated Yelp database (see downloads)
└── queries
    └── geo	# See experiments data
        ├── geo_qdmr_train.json
	└── geo_qdmr_predicted_train.json
	└── geo_gold_train.json
	└── geo_gold_dev.json
	└── geo_gold_test.json
	└── geo_gold_train.sql
	└── geo_gold_dev.sql
	└── geo_gold_test.sql
    └── spider
        ├── spider_qdmr_train.json		# See experiments data
	└── spider_qdmr_predicted_train.json 	# See experiments data
	└── spider_gold_train.json 	# Spider training set
	└── spider_gold_dev.json 	# Spider dev set
	└── spider_gold_train.sql 	# Spider training set SQL queries
	└── spider_gold_dev.sql 	# Spider dev set SQL queries
```
Database files are described in the downloads section. See the experiments section for the exact train and test files.

#### 2. Train model
To train the text-to-SQL model configure its following parameters in `train.py`:
* `dataset`: either `spider` or `geo`
* `target_encoding`: `sql` for gold sql and either `qdmr_formula` or `qdmr_sql` for the QDMR experiments
* `data_dir`: path to the directory containing the experiments data
* `output_dir`: the directory to store the trained model
* `db_dir`: the directory to store the trained model
* `training_set_file`: training set file in the data directory e.g. `spider/spider_gold_train.json`
* `dev_set_file`: dev set file in the data directory e.g. `spider/spider_gold_dev.json`
* `dev_set_sql`: dev set SQL queries in the data directory e.g. `spider/spider_gold_dev.sql`

Following configuration, to train the model run:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py 
```

#### 3. Test model

To test the text-to-SQL model first configure the relevant parameters and `checkpoint_path` in `test.py`.
Following the configuration, generate the trained model predictions using:
```bash
CUDA_VISIBLE_DEVICES=0 python test.py 
```


## Experiments ⚗️

### Data

#### Gold SQL:
For the Spider experiments we use its original train and dev `json` and `sql` files. For Geo880, Academic, IMDB and Yelp we format the original datasets in `json` files available [**here**](https://github.com/tomerwolgithub/question-decomposition-to-sql/blob/main/data/text_to_sql/gold_sql_datasets.zip).

#### QDMR Synthesized SQL:
The QDMR text-to-SQL models are not trained directly on the synthesized SQL. Instead, we train on an encoded QDMR representation with its phrase-DB linking (from the SQL synthesis). This representation is automatically mapped to SQL to evaluate the models execution accuracy.
To generate these *grounded QDMRs* we use the output of the data generation phase. The function `encoded_grounded_qdmr` in `src/data_generation/write_encoding.py` recieves the `json` file containing the synthesized SQL examples. It then encodes them as lisp style formulas of QDMR steps and their relevant phrase-DB linking.

For convenience, you can download the encoded QDMR training sets used in our experiments [**here**](https://github.com/tomerwolgithub/question-decomposition-to-sql/blob/main/data/text_to_sql/encoded_qdmr_datasets.zip). These include:

 * `qdmr_ground_enc_spider_train.json`: 5,349 examples, synthesized using gold QDMR + answer supervision
 * `qdmr_ground_enc_predicted_spider_train_few_shot`: 5,075 examples, synthesized examples using 700 gold QDMRs, predicted QDMR + answer supervision
 * `qdmr_ground_enc_predicted_spider_train_30_db.json`: 1,129 examples, synthesized using predicted QDMR + answer supervision
 * `qdmr_ground_enc_predicted_spider_train_40_db.json`: 1,440 examples, synthesized using predicted QDMR + answer supervision
 * `qdmr_ground_enc_predicted_spider_train_40_db_V2.json`: 1,552 examples, synthesized using predicted QDMR + answer supervision
 * `qdmr_ground_enc_geo880_train.json`: 454 examples, synthesized using gold QDMR + answer supervision
 * `qdmr_ground_enc_predicted_geo_train_zero_shot.json`: 432 examples, synthesized using predicted QDMR + answer supervision


### Configurations

The configurations for training the text-to-SQL models on **Spider**. Other parameters are fixed in `train.py`.

**SQL Gold (Spider):**
```bash
{'dataset': 'spider',
'target_encoding': 'sql',
'db_dir': 'databases/spider_databases',
'training_set_file': 'queries/spider/spider_gold_train.json',
'dev_set_file': 'queries/spider/spider_gold_dev.json',
'dev_set_sql': 'queries/spider/spider_gold_dev.sql'}
```

**QDMR Gold (Spider):**
```bash
{'dataset': 'spider',
'target_encoding': 'qdmr_formula',
'db_dir': 'databases/spider_databases',
'training_set_file': 'queries/spider/spider_qdmr_train.json',
'dev_set_file': 'queries/spider/spider_gold_dev.json',
'dev_set_sql': 'queries/spider/spider_gold_dev.sql'}
```

**SQL Predicted (Spider):**
```bash
{'dataset': 'spider',
'target_encoding': 'qdmr_formula',
'db_dir': `databases/spider_databases',
'training_set_file': 'queries/spider/spider_qdmr_predicted_train.json',
'dev_set_file': 'queries/spider/spider_gold_dev.json',
'dev_set_sql': 'queries/spider/spider_gold_dev.sql'}
```

The configurations for training the text-to-SQL models on **Geo880**.

**SQL Gold (Geo):**
```bash
{'dataset': 'geo',
'target_encoding': 'sql',
'db_dir': 'databases',
'training_set_file': 'queries/geo/geo_gold_train.json',
'dev_set_file': 'queries/spider/geo_gold_dev.json',
'dev_set_sql': 'queries/spider/geo_gold_dev.sql'}
```

**QDMR Gold (Geo):**
```bash
{'dataset': 'geo',
'target_encoding': 'qdmr_sql',
'db_dir': 'databases',
'training_set_file': 'queries/geo/geo_qdmr_train.json',
'dev_set_file': 'queries/spider/geo_gold_dev.json',
'dev_set_sql': 'queries/spider/geo_gold_dev.sql'}
```

**QDMR Predicted (Geo):**
```bash
{'dataset': 'geo',
'target_encoding': 'qdmr_sql',
'db_dir': 'databases',
'training_set_file': 'queries/geo/geo_qdmr_predicted_train.json',
'dev_set_file': 'queries/spider/geo_gold_dev.json',
'dev_set_sql': 'queries/spider/geo_gold_dev.sql'}
```

### Evaluation
Text-to-SQL model performance is evaluated using SQL execution accuracy in [`src/text_to_sql/eval_spider.py`](https://github.com/tomerwolgithub/question-decomposition-to-sql/blob/main/src/text_to_sql/eval_spider.py).
The script automatically converts encoded QDMR predictions to SQL before executing them on the target database.


## Citation ✍🏽

```
bibtex
@inproceedings{wolfson-etal-2022-weakly,
    title={"Weakly Supervised Text-to-SQL Parsing through Question Decomposition"},
    author={"Wolfson, Tomer and Deutch, Daniel and Berant, Jonathan"},
    booktitle = {"Findings of the Association for Computational Linguistics: NAACL 2022"},
    year={"2022"},
}
```

## License 
This repository and its data is released under the MIT license.

For the licensing of all external datasets and databases used throughout our experiments:
* The Break dataset [(Wolfson et al., 2020)](https://allenai.github.io/Break/) is under the MIT License. 
* Spider [(Yu et al., 2018)](https://yale-lily.github.io/spider) is under the CC BY-SA 4.0 License. 
* Geo880 [(Zelle and Mooney, 1996)](https://www.aaai.org/Library/AAAI/1996/aaai96-156.php) is available under the GNU General Public License 2.0.
* The text-to-SQL versions of Academic [(Li and Jagadish, 2014)](https://www.vldb.org/pvldb/vol8/p73-li.pdf) and Geo880 were made publicly available by [Finegan-Dollak et al. (2018)](https://github.com/jkkummerfeld/text2sql-data/).
* The IMDB and YELP datasets were publicly released by [Yaghmazadeh et al. (2017)](https://drive.google.com/drive/folders/0B-2uoWxAwJGKY09kaEtTZU1nTWM).
