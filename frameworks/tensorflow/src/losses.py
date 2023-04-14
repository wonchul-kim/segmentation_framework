
import frameworks.tensorflow.models as tf_models

def get_criterion(loss_fn, focal_loss, class_weights):
    if loss_fn == 'dice':
        criterion = tf_models.losses.DiceLoss(class_weights=class_weights)
    
    if focal_loss:
        criterion += tf_models.losses.CategoricalFocalLoss(alpha=0.25, gamma=2.0) 
        
        
    iou_score = tf_models.metrics.IOUScore(threshold=0.5)
        
    return criterion, iou_score