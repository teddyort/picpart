clear, clc
label = imread('../models/label_1.png');
pred = imread('../models/prediction_1.png');
figure(1)
imagesc(label)
figure(2)
imagesc(pred)
acc = sum(sum(label == pred))/numel(label);
fprintf('accuracy: %.3f\n',acc)
[area_intersection, area_union] = intersectionAndUnion(pred,label,150);
IoU = area_intersection./(eps+area_union);
meanIoU = mean(IoU);
fprintf('meanIoU: %.3f\n',meanIoU)
% image 1
% mean pixel accuracy: 0.743
% mean IoU: 0.011828515002905554

% image 2
% mean pixel accuracy: 0.633
% mean IoU: 0.026138981379926953

