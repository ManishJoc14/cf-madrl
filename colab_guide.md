# Running CF-MADRL on Google Colab

## 1. Prepare Your Project

Zip using:

```python
python .\pack_for_colab.py
```

## 2. Upload to Colab

Select T4 GPU for faster training.
Upload `traffic_cf_madrl_colab.zip` to the session storage.

## 3. Setup Environment (Run these cells)

**Cell 1: Unzip**

```python
!unzip /content/traffic_cf_madrl_colab.zip
```

**Cell 2: Run Setup Script**

```python
!chmod +x colab_setup.sh
!./colab_setup.sh
```

## 4. Run Training

**Cell 3: Start Training**
Ensure 'gui: false' is in config.yaml in all places.
Enter the desired number of training episodes (e.g., 20):

```python
!python main.py --mode train 20
```

## 5. Run Evaluation

**Cell 4: Start Evaluation**

```python
!python main.py --mode eval
```

## 6. Download Results

**Cell 5: Zip Results**

```python
!zip -r output.zip logs plots saved_models
```

**Cell 6: Download Output**

```python
from google.colab import files
files.download('output.zip')
```
