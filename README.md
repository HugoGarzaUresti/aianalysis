# aianalysis

Data Analysis web app capable of transforming statistical data into meaningful insights through machine learning algorithms.

Proyecto integrador

## Setup

Clone the repo

### Back-end (Flask)

In the terminal `cd` into the project (`cd aianalysis`). Then, do the following steps:

1. Create a python virtual environment

Syntax:

```sh
python3 -m venv <NAME_OF_VIRTUAL_ENV>
```

Example:

```sh
python3 -m venv venv
```

> **Note**: In case you choose another name for your virtual env, make sure to put the name in the `.gitignore` file located in the `aianalysis` folder

2. Activate virtual environment

```sh
source <NAME_OF_VIRTUAL_ENV>/bin/activate
```

Example:

If the name of your virtual env is `venv`, then you should run:

```sh
source venv/bin/activate
```

> **Note**: Make sure you are using the virtual environment by running `which python` in the terminal. It should display something like `/YOUR_PATH/aianalysis/venv/bin/python`
>
> (If you have conda `base` environment you may want to deactive that environment first by doing `conda deactivate`)

3. Install python dependencies

```sh
pip install -r requirements.txt
```

4. Start server

```sh
flask run
```

### Front-end (React)

Move into the frontend folder `cd frontend`

1. Install dependencies


```sh
npm install
```

2. Start server

```sh
npm run start
```
