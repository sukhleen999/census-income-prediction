# Docker file for the census income prediction
# Author: Philson
# December, 2021

# Use anaconda as the base image
FROM continuumio/anaconda3

# Install System Pre-requisites
RUN apt update && \
    apt install -y software-properties-common build-essential libcurl4-openssl-dev libssl-dev libxml2-dev

# Install R (Version 4.1.2)
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-key '95C0FAF38DB3CCAD0C080A7BDC78B2DDEABC47B7'

RUN add-apt-repository "deb http://cloud.r-project.org/bin/linux/debian bullseye-cran40/" && \
    apt update && \
    apt install -y -t bullseye-cran40 r-base r-base-dev

# Install required packages in R
RUN Rscript -e "install.packages('tidyverse');"


# Download the conda environment file and create conda environment
RUN cd home && \
    wget https://raw.githubusercontent.com/UBC-MDS/census-income-prediction/main/census-income.yaml && \
    conda env create -f census-income.yaml

# Garbage collection
RUN rm /home/census-income.yaml 
