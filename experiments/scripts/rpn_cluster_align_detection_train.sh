#!/usr/bin/env bash
job_id=$1
config_file=$2

if [ $HOME == '/' ]; then
  if [ -d "/ghome/zhangyx" ]; then
    export HOME=/ghome/zhangyx
    echo 'HOME is '${HOME}
    cd '/ghome/zhangyx/PycharmProjects/RPA' || exit
    data_root=/ghome/zhangyx/PycharmProjects/semseg/data
  fi
elif [ $HOME == '/root' ]; then
  export HOME='/code'
  data_root=/code/data
  cd $HOME || exit
else
  cd $HOME'/PycharmProjects/RPA' || exit
  data_root=/home/zhangyx/PycharmProjects/RPA/data
fi

trainer_class=rpnclusteralign
python_file=./training_scripts.py
python ${python_file} --task_type det --job_id ${job_id} --config ${config_file} --trainer ${trainer_class} --data_root ${data_root}
