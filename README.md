# Deep Recurrent Actor-Critic based Service Migration

## OS requirement 

- OS: Unbuntu 18.04 or Windows 10


## Configure the virtual environment

It is better to use a virtual environment (e.g., Anaconda) to run the code. About how to install anaconda, please refer to the official website: https://www.anaconda.com

- Python version: >= 3.8
- Tensorflow >= 2.0

For other related third-party libraries, please refer to `pomdp-service-migration.yaml`

## Running the Code

To run the training of all involved algorithms, just simply run the code:

``` python
# train the dracm with rome traces
python training_with_rome_races.py
# train the dracm with sanfrancisco traces
python training_with_san_traces.py 

# running the training of all baseline algorithms.
.....

# testing the dracm with different scenarioes
python test_different_arriving_rate_rome.py
python test_diiferent_migration_cost_rome.py

....
```

The training may take long time, please be patient.

## Related publication

[Online Service Migration in Mobile Edge with Incomplete System Information: A Deep Recurrent Actor-Critic Learning Approach](https://arxiv.org/abs/2012.08679)

If you are interested with this work, please cite the paper:

```tex
@article{wang2022online,
  author={Wang, Jin and Hu, Jia and Min, Geyong and Ni, Qiang and El-Ghazawi, Tarek},
  journal={IEEE Transactions on Mobile Computing}, 
  title={Online Service Migration in Mobile Edge with Incomplete System Information: A Deep Recurrent Actor-Critic Learning Approach}, 
  year={2022},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TMC.2022.3197706}}
```


