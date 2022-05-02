P_TRAIN=0
P_TEST_LO=0
P_TEST_HI=2
seeds=(0)
gammas=(1.25 1.5 1.75 2)

for SEED in "${seeds[@]}";
do
    python train.py main --n_train 12000 --dataset mimic --p_test_lo $P_TEST_LO --p_test_hi $P_TEST_HI --p_train $P_TRAIN --n_test_sweep 4 --method erm --epochs 25 --loss squared_loss --seed $SEED --save mimic --unobserved "age_on_adm"
  # Standard ERM
    python train.py main --n_train 12000 --dataset mimic --p_test_lo $P_TEST_LO --p_test_hi $P_TEST_HI --p_train $P_TRAIN --n_test_sweep 4 --method erm --epochs 25 --loss squared_loss --seed $SEED --save mimic --unobserved "los"
  # RU Regression

   # RU Regression
    for GAMMA in "${gammas[@]}";
    do
        python train.py main --n_train 12000 --dataset mimic --p_test_lo $P_TEST_LO --p_test_hi $P_TEST_HI --p_train $P_TRAIN --n_test_sweep 4 --method ru_regression --epochs 25 --loss squared_loss --gamma $GAMMA --seed $SEED --save mimic --unobserved "age_on_adm"
    done


   for GAMMA in "${gammas[@]}";
   do
        python train.py main --n_train 12000 --dataset mimic --p_test_lo $P_TEST_LO --p_test_hi $P_TEST_HI --p_train $P_TRAIN --n_test_sweep 4 --method ru_regression --epochs 25 --loss squared_loss --gamma $GAMMA --seed $SEED --save mimic --unobserved "los"
   done
  # Standard ERM
done


