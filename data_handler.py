import LoggingTools.bucket_handler as bh
import io


upload_dataframe_csv(dataframe, bucket, blob_path):
    b_buffer = io.BytesIO() 
    dataframe.to_csv(b_buffer, index=True)
    bh.upload_to_bucket(bucket, blob_path, b_buffer)

upload_metric(figure, bucket, blob_path):
    b_buffer = io.BytesIO()
    fig.savefig(b_buffer, format="png")
    bh.upload_to_bucket(bucket, blob_path, b_buffer)

