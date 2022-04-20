# Weakly Supervised Text-to-SQL Parsing through Question Decomposition
The official repository for the Finings of NAACL 2022 paper, titled "Weakly Supervised Text-to-SQL Parsing through Question Decomposition" by Tomer Wolfson, Daniel Deutch and Jonathan Berant.

This repository contains the code and data used in our paper:

1. Code for automatically synthesizing SQL queries from question decompositions + answers
2. Code for the models used in our paper mapping text-to-SQL and text-to-QDMR 

* **Structure:**
	* [**Setup**](https://)
	* [**Resources**](https://)
	* [**Data Generation**](https://)
	* [**Models**](https://)
	* [**Experiments**](https://)
	* [**Citation**](https://)
	* [**License**](https://)


## Setup

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

### QDMR Parsing Datasets:
* Academic, GeoQuery and Spider QDMRs found in the Break dataset: [https://allenai.github.io/Break](https://allenai.github.io/Break)
* QDMR data for IMDB and Yelp is available [here](/data/qdmr_annotation)

### Text-to-SQL Datasets:
* [Spider](https://yale-lily.github.io/spider)
* [Academic, IMDB, Yelp and GeoQuery](https://github.com/jkkummerfeld/text2sql-data/tree/master/data/original) (Jonathan Kummerfeld's repository)

### Databases (schema & contents):
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

1. The SQL synthesis expect a formatted `csv` file, see [example](). Note that gold SQL is only used for computing the gold answer.
2. This may take several hours, as multiple candidate SQL are executed on their respective database.
3. Synthesize SQL from the `<QDMR, database, answer>` examples:

```bash
python data_generation/main.py \
--input_file input_qdmr_examples.csv \
--output_file qdmr_grounded_sql.csv \
--json_steps True
```

### Synthesized Data
The evaluation sets generated with BPB for the development sets of DROP, HotpotQA, and IIRC are available for download under the data directory (one file per dataset). Each zip file includes the following files:
The SQL synthesized using QDMR + answer supervision is available for each dataset in the `data/sql_synthesis_results/ directory. 
* `data/sql_synthesis_results/gold_qdmr_supervision`: contains SQL synthesized using gold QDMR that was manually annotated
* `data/sql_synthesis_results/predicted_qdmr_supervision`: contains SQL synthesized using QDMR predicted by a trained parser




## Models 🗂️

### Text-to-SQL 
### Text-to-QDMR
### QDMR Parser

### Data Setup

Create a directory containing all relevant databases in sqlite format:
```
cd question-decomposition-to-sql
mkdir databases
```

## Experiments ⚗️

### Data
### Evaluation

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
For the external datasets and databases used throughput our experiments:
* The Break dataset [(Wolfson et al., 2020)](https://allenai.github.io/Break/) is under the MIT License. 
* Spider [(Yu et al., 2018)](https://yale-lily.github.io/spider) is under the CC BY-SA 4.0 License. 
* Geo880 [(Zelle and Mooney, 1996)](https://www.aaai.org/Library/AAAI/1996/aaai96-156.php) is available under the GNU General Public License 2.0.
* The text-to-SQL versions of Academic [(Li and Jagadish, 2014)](https://www.vldb.org/pvldb/vol8/p73-li.pdf) and Geo880 were made publicly available by [Finegan-Dollak et al. (2018)](https://github.com/jkkummerfeld/text2sql-data/).
* The IMDB and YELP datasets were publicly released by [Yaghmazadeh et al. (2017)](goo.gl/DbUBMM).
