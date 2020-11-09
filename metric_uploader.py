import LoggingTools.general_metrics as gm
import LoggingTools.data_handler as dh
import LoggingTools.logging as lg
import random

bucket = "training_checkpoint_data"

def upload_heatmap_metric(pred_data, target_data, model_name, timestamp, epoch):
    figure = gm.heatmap_avg_distance(pred_data, target_data)
    
    dh.upload_metric(figure, bucket, "/"+model_name+"/"+str(timestamp)+"/heatmaps/overall/"+str(epoch)+"-hm.png")

def upload_heatmap_metric_single(pred_data, target_data, model_name, timestamp, epoch):
    
    batches_num = target_data.shape[0]
    index = random.range(batches_num)
    figure = gm.heatmap_avg_distance_sigle_signal(pred_data[index:index+1], target_data[index:index+1])
    
    dh.upload_metric(figure, bucket, "/"+model_name+"/"+str(timestamp)+"/heatmaps/single/"+str(epoch)+"-hm.png")

def upload_log(pred_data, target_data, model_name, timestamp, epoch, logger):

    df = logger.data_frame
    figure = logger.create_lineplot()

    dh.upload_log(df, bucket, "/"+model_name+"/"+str(timestamp)+"/dataframe/df.csv")
    dh.upload_metric(figure, bucket, "/"+model_name+"/"+str(timestamp)+"/metrics/mt.png")