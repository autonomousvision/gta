# RealEstate10K and ACID Downloaders
These scripts are used to download RealEstate10K dataset. 

## How to use   
First, you should download the [RealEstate10K](https://google.github.io/realestate10k/download.html) and [ACID](https://storage.googleapis.com/gresearch/aerial-coastline-imagery-dataset/acid_large_v1_release.tar.gz) poses. The RealEstate10k train/test/validation poses should be stored in the subfolder RealEstate10K/ and the ACID train/test/validation poses should be stored in a subfolder ACID/

Next run the following command.

```shell
python3 generate_realestate.py [test or train]
python3 generate_acid.py [test, validation or train]
```
This downloads YouTube movies and extract frames which are needed.  Because of unkown reasons, `pytube` fails to download and save movies. 
In this case, sequence name is added to `failed_videos_{test, train}.txt`.   

The scripts used here are based on the scripts [here](https://github.com/cashiwamochi/RealEstate10K_Downloader)
