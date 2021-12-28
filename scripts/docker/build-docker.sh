
# run this file from the scripts/docker directory
cp -R * ../../..
cd ../../..

# build 1
docker build -t moseq2 .

# build 2 - smaller, needs more testing, might be more risky
docker build -t moseq2-multistage -f mamba-multistage/Dockerfile .
