## train_1.sh

cd ../src
python -u fnn.py --config config.yaml --batch_size 64 --optimizer adam > ../runtime_log/train_1.log 2>&1