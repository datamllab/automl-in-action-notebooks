cp ../../requirements.txt ./
docker build . -f Dockerfile -t book:latest
rm requirements.txt
