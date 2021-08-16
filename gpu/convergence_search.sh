for learning_rate in 0.01 0.005 0.001 0.0005 0.0001; do
    for n_iter_test in 200 300 400 500 600; do
        echo "python3 mem_comprimize.py $learning_rate $n_iter_test" >> convergence_search.txt
        python3 mem_compromize.py $learning_rate $n_iter_test
        python3 convergence_check.py >> convergence_search.txt
    done;
done;
