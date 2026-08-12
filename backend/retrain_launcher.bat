@echo off
cd /d "D:\Projects\Retinasense Ai\backend"
"C:\Users\nvdah\AppData\Local\Programs\Python\Python313\python.exe" -u train_v2.py --batch-size 24 > retrain_me.log 2> retrain_me.err.log
