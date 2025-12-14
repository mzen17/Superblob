docker run --rm -p 1000:1000 \
  --add-host host.docker.internal:host-gateway \
  -v $(pwd)/nginx.conf:/etc/nginx/nginx.conf:ro \
  -v $(pwd)/surveyapp/build:/surveyapp/build:ro \
  nginx
