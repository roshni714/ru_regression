seeds=(0)
gammas=(1.5 2 2.5 3)

for SEED in "${seeds[@]}";
do
  #Standard ERM
     python train.py main --dataset mimic --p_test_lo $P_TEST_LO --p_test_hi $P_TEST_HI_LOS --p_train $P_TRAIN --n_test_sweep 2 --method erm --epochs 10 --loss squared_loss --seed $SEED --save mimic_los --unobserved "los"
  # RU Regression
    for GAMMA in "${gammas[@]}";
    do
        python train.py main --dataset mimic --p_test_lo $P_TEST_LO --p_test_hi $P_TEST_HI_LOS --p_train $P_TRAIN --n_test_sweep 2 --method ru_regression --epochs 10 --loss squared_loss --gamma $GAMMA --seed $SEED --save mimic_los --unobserved "los"
   done

done


