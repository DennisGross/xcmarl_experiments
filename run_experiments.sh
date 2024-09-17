#source venv/bin/activate
python normal_train.py --samples=2 --training_steps=1000 --number_of_tests=3
python modified_obs_train.py --samples=2 --training_steps=1000 --number_of_tests=3