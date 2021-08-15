for learning_rate in 0.01 0.005 0.001; do
    for n_iter_train in 1000 2000 3000; do
        echo "python3 mem_comprimize.py $learning_rate $n_iter_train" >> convergence_search.txt
        python3 mem_compromize.py $learning_rate $n_iter_train
        python3 convergence_check.py >> convergence_search.txt
    done;
done;