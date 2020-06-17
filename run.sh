GPU_ID=$1
DATASET=$2
EPOCHS=${3:-30}
VOCAB_SIZE=${4:-5000}
INPUT_SIZE=${5:-300}
HIDDEN_SIZE=${6:-100}
ELAPSED_TIME=${7:-3000}
TWEETS_COUNT=${8:-500}
SEED=${9:-0}
LR=${10:-'0.005'}
WEIGHT_DECAY=${11:-'5e-4'}
echo "gpu_id:${GPU_ID} dataset:${DATASET} epochs:${EPOCHS} vocab_size:${VOCAB_SIZE} input_size:${INPUT_SIZE} hidden_size:${HIDDEN_SIZE} elapsed_time:${ELAPSED_TIME} tweets_count:${TWEETS_COUNT} seed:${SEED} lr:${LR} weight_decay:${WEIGHT_DECAY}"
#LOG="logs/${DATASET}_f${FOLD_IDX}_100_w4_s_uatt.log"
#exec &> >(tee -a "$LOG")
#echo Logging output to "$LOG"

CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py --dataset ${DATASET} --epochs ${EPOCHS} --vocab_size ${VOCAB_SIZE} --input_size ${INPUT_SIZE} --hidden_size ${HIDDEN_SIZE} --elapsed_time ${ELAPSED_TIME} --tweets_count ${TWEETS_COUNT} --seed ${SEED} --lr ${LR} --weight_decay ${WEIGHT_DECAY} --filename result/${DATASET}_et${ELAPSED_TIME}_tc${TWEETS_COUNT}.txt
