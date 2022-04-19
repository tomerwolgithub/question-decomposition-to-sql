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
	* [**Evaluation**](https://)
	* [**Citation**](https://)

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

## Download Resources

1. Download latest version of Spider's sqlite databases: [https://yale-lily.github.io/spider](https://yale-lily.github.io/spider)

2. Download other databases:
	* Academic, IMDB & Yelp (Yaghmazadeh et al., OOPSLA 2017): [https://drive.google.com/drive/folders/0B-2uoWxAwJGKY09kaEtTZU1nTWM](https://drive.google.com/drive/folders/0B-2uoWxAwJGKY09kaEtTZU1nTWM)
	* GeoQuery (Zelle & Mooney, AAAI 1996): [https://github.com/jkkummerfeld/text2sql-data](https://github.com/jkkummerfeld/text2sql-data)

3. Convert Academic, IMDB, Yelp and GeoQuery to sqlite format using the tool by [Jean-Luc Lacroix](https://github.com/dumblob/mysql2sqlite)

4. Create a directory containing all relevant databases in sqlite format


## Citation ✍🏽

```bibtex
@inproceedings{wolfson-etal-2022-weakly,
    title={"Weakly Supervised Text-to-SQL Parsing through Question Decomposition"},
    author={"Wolfson, Tomer and Deutch, Daniel and Berant, Jonathan"},
    booktitle = {"Findings of the Association for Computational Linguistics: NAACL 2022"},
    year={"2022"},
}
```

