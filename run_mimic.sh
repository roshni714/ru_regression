P_TRAIN=0
P_TEST_LO=0
P_TEST_HI_AGE=1
P_TEST_HI_LOS=1
P_TEST_HI_GENDER=1
P_TEST_HI_ETHNICITY=1
seeds=(0)
gammas=(1.5 2 2.5 3)

for SEED in "${seeds[@]}";
do
     python train.py main --dataset mimic --p_test_lo $P_TEST_LO --p_test_hi $P_TEST_HI_LOS --p_train $P_TRAIN --n_test_sweep 2 --method erm --epochs 10 --loss squared_loss --seed $SEED --save mimic_los --unobserved "los"
#    python train.py main --dataset mimic --p_test_lo $P_TEST_LO --p_test_hi $P_TEST_HI_GENDER --p_train $P_TRAIN --n_test_sweep 2 --method erm --epochs 10 --loss squared_loss --seed $SEED --save mimic_gender --unobserved "gender"
#     python train.py main --dataset mimic --p_test_lo $P_TEST_LO --p_test_hi $P_TEST_HI_AGE --p_train $P_TRAIN --n_test_sweep 2 --method erm --epochs 10 --loss squared_loss --seed $SEED --save mimic_age --unobserved "age_on_adm"
#    python train.py main --dataset mimic --p_test_lo $P_TEST_LO --p_test_hi $P_TEST_HI_ETHNICITY --p_train $P_TRAIN --n_test_sweep 2 --method erm --epochs 10 --loss squared_loss --seed $SEED --save mimic_ethnicity --unobserved "ethnicity"

  # RU Regression

  # RU Regression
    for GAMMA in "${gammas[@]}";
    do
#        python train.py main --dataset mimic --p_test_lo $P_TEST_LO --p_test_hi $P_TEST_HI_AGE --p_train $P_TRAIN --n_test_sweep 2 --method ru_regression --epochs 10 --loss squared_loss --gamma $GAMMA --seed $SEED --save mimic_age --unobserved "age_on_adm"
#        python train.py main --dataset mimic --p_test_lo $P_TEST_LO --p_test_hi $P_TEST_HI_GENDER --p_train $P_TRAIN --n_test_sweep 2 --method ru_regression --epochs 10 --loss squared_loss --gamma $GAMMA --seed $SEED --save mimic_gender --unobserved "gender"
        python train.py main --dataset mimic --p_test_lo $P_TEST_LO --p_test_hi $P_TEST_HI_LOS --p_train $P_TRAIN --n_test_sweep 2 --method ru_regression --epochs 10 --loss squared_loss --gamma $GAMMA --seed $SEED --save mimic_los --unobserved "los"
#        python train.py main --dataset mimic --p_test_lo $P_TEST_LO --p_test_hi $P_TEST_HI_ETHNICITY --p_train $P_TRAIN --n_test_sweep 2 --method ru_regression --epochs 10 --loss squared_loss --gamma $GAMMA --seed $SEED --save mimic_ethnicity --unobserved "ethnicity"
   done

  # Standard ERM
done


