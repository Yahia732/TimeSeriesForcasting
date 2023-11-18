The lags folder contains the number of lags each dataset needs make sure then you send at least that number.
if there's nan values make sure there is at least non nan values that is equal to the number of lags needed.
to run the model either you simply run the app on your ide or you can run the docker compose
if you chose docker compose open port 8000 to run the application after that send post request to the endpoint that is written before the build_model function and you will get the response with the prediction
