import marimo

__generated_with = "0.19.9"
app = marimo.App()


@app.cell
def _():
    import os

    files = os.listdir("/home/alphabridge/PythonProjects/coliee2026/data/task1/task1_train_files_2025/")
    return (files,)


@app.cell
def _(files):
    files
    return


@app.cell
def _():
    with open("/home/alphabridge/PythonProjects/coliee2026/data/task1/task1_train_files_2025/086870.txt", 'r') as f:
        # sample_txt = f.read()
        sample_txt_lines = f.readlines()
    return (sample_txt_lines,)


@app.cell
def _(sample_txt_lines):
    sample_txt_lines
    return


if __name__ == "__main__":
    app.run()
