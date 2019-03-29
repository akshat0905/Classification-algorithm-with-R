#For finding out the best predictions in a multi-class classification in such a case when
#where we want to build a model for predicting people who were readmitted in less than 30 days



library(class)
library(caret)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(randomForest)
library(e1071)
set.seed(123)



path="C:\\Users\\Admin\\Documents\\R Programming\\Project_mock_r\\dataset_diabetes\\diabetic_data.csv"

diab=read.csv(path,header=T)



table(diab$glimepiride.pioglitazone)
table(diab$metformin.rosiglitazone)
table(diab$metformin.pioglitazone)
# Performing EDA

col_name = colnames(diab) [apply(diab, 2, function(n) any(is.na(n)))]
if(length(col_name) > 0) print("NA's present") else print("No NA's")
print(col_name)

col_name = colnames(diab) [apply(diab, 2, function(n) any(n == ""))]
if(length(col_name) > 0) print("Blanks present") else print("No Blanks")

col_name = colnames(diab) [apply(diab, 2, function(n) any(n=='?'))]
print(col_name)


plot(diab$gender, main = "gender distribution")

plot(diab$race, main="race")

#fixing missing values

levels(diab$race)[levels(diab$race)=="?"] <- "Unk" # converted missing values to other



levels(diab$gender)[levels(diab$gender)=="Unknown/Invalid"] = "Female" # low number of unknown/Invalid so converted to mode

#reducing level based on IDS mapping
diab$admission_type_id=as.factor(diab$admission_type_id)
levels(diab$admission_type_id)[levels(diab$admission_type_id)=='6' | levels(diab$admission_type_id)=='8']= '5'
levels(diab$admission_type_id)[levels(diab$admission_type_id)=='1' | levels(diab$admission_type_id)=='2' | levels(diab$admission_type_id)=='4']= '7'




diab$admission_source_id=as.factor(diab$admission_source_id)
#diab$time_in_hospital=as.factor(diab$time_in_hospital) # converted it to factor variable because of only 14 values present
levels(diab$admission_source_id)[levels(diab$admission_source_id)=='15' | levels(diab$admission_source_id)=='17' | levels(diab$admission_source_id)=='20' | levels(diab$admission_source_id)=='21']='9'
levels(diab$admission_source_id)[levels(diab$admission_source_id)=='2' | levels(diab$admission_source_id)=='3']='1'
levels(diab$admission_source_id)[levels(diab$admission_source_id)=='11' | levels(diab$admission_source_id)=='23' | levels(diab$admission_source_id)=='24']='8'
levels(diab$admission_source_id)[levels(diab$admission_source_id)=='12' | levels(diab$admission_source_id)=='13' | levels(diab$admission_source_id)=='14']='7'
levels(diab$admission_source_id)[levels(diab$admission_source_id)!='1' & levels(diab$admission_source_id)!='8' & levels(diab$admission_source_id)!='7'& levels(diab$admission_source_id)!='9']='4'

table(diab$admission_source_id)


diab$discharge_disposition_id=as.factor(diab$discharge_disposition_id)
levels(diab$discharge_disposition_id)[levels(diab$discharge_disposition_id)=='13']='1'
levels(diab$discharge_disposition_id)[levels(diab$discharge_disposition_id) %in% c('19','20','21')]='11'

levels(diab$discharge_disposition_id)[levels(diab$discharge_disposition_id) %in% c('25','26')]='18'
levels(diab$discharge_disposition_id)[levels(diab$discharge_disposition_id) %in% c('3','4','5','6','8','12','15','10','14','16','17','22','23','24','30','27','28','29')]='2'
table(diab$discharge_disposition_id)


levels(diab$medical_specialty)
str(diab$num_lab_procedures)
str(diab$num_medications)
str(diab$num_procedures)
100*prop.table(table(diab$medical_specialty))


str(diab)
str(diab$num_medications)
# removing columns which are not required
diab$encounter_id = NULL
diab$patient_nbr = NULL
#diab$weight = NULL
#diab$payer_code = NULL
#diab$medical_specialty = NULL
diab$citoglipton = NULL
diab$examide = NULL

table(diab$citoglipton)

table(diab$examide)

ncol(diab)


str(diab)
cor(diab[8:13])

#Diagnosis 1

table(diab$diag_1)#Since It has too many Variable we will Group the variable
levels(diab$diag_1)[levels(diab$diag_1) %in% c(390:459, 785)] <- "Circulatory"
levels(diab$diag_1)[levels(diab$diag_1) %in% c(460:519, 786)] <- "Respiratory"
levels(diab$diag_1)[levels(diab$diag_1) %in% c(520:579, 787)] <- "Digestive"
levels(diab$diag_1)[levels(diab$diag_1) %in% c(seq.default(from = 250,to = 250.99,by =0.01))] <- "Diabetes"
levels(diab$diag_1)[levels(diab$diag_1) %in% c(800:999)] <- "Injury"
levels(diab$diag_1)[levels(diab$diag_1) %in% c(710:739)] <- "Musculoskeletal"
levels(diab$diag_1)[levels(diab$diag_1) %in% c(580:629,788)] <- "Genitourinary"
levels(diab$diag_1)[levels(diab$diag_1) %in% c(140:239,780,781,784,790:799,240:249,251:279,680:709,782,001:139)] <- "Neoplasms"
Defined=c("Circulatory","Respiratory","Digestive","Diabetes","Injury","Musculoskeletal","Genitourinary","Neoplasms")
levels(diab$diag_1)[!(levels(diab$diag_1) %in% Defined)] <- "Other"
table(diab$diag_1)#Grouped levels by ICD9 codes
#Diagnosis 2
table(diab$diag_2)#Since It has too many Variable we will Group the variable
levels(diab$diag_2)[levels(diab$diag_2) %in% c(390:459, 785)] <- "Circulatory"
levels(diab$diag_2)[levels(diab$diag_2) %in% c(460:519, 786)] <- "Respiratory"
levels(diab$diag_2)[levels(diab$diag_2) %in% c(520:579, 787)] <- "Digestive"
levels(diab$diag_2)[levels(diab$diag_2) %in% c(seq.default(from = 250,to = 250.99,by =0.01))] <- "Diabetes"
levels(diab$diag_2)[levels(diab$diag_2) %in% c(800:999)] <- "Injury"
levels(diab$diag_2)[levels(diab$diag_2) %in% c(710:739)] <- "Musculoskeletal"
levels(diab$diag_2)[levels(diab$diag_2) %in% c(580:629,788)] <- "Genitourinary"
levels(diab$diag_2)[levels(diab$diag_2) %in% c(140:239,780,781,784,790:799,240:249,251:279,680:709,782,001:139)] <- "Neoplasms"
Defined=c("Circulatory","Respiratory","Digestive","Diabetes","Injury","Musculoskeletal","Genitourinary","Neoplasms")
levels(diab$diag_2)[!(levels(diab$diag_2) %in% Defined)] <- "Other"
table(diab$diag_2)#Grouped levels by ICD9 codes
#Diagnosis 3
table(diab$diag_3)#Since It has too many Variable we will Group the variable
levels(diab$diag_3)[levels(diab$diag_3) %in% c(390:459, 785)] <- "Circulatory"
levels(diab$diag_3)[levels(diab$diag_3) %in% c(460:519, 786)] <- "Respiratory"
levels(diab$diag_3)[levels(diab$diag_3) %in% c(520:579, 787)] <- "Digestive"
levels(diab$diag_3)[levels(diab$diag_3) %in% c(seq.default(from = 250,to = 250.99,by =0.01))] <- "Diabetes"
levels(diab$diag_3)[levels(diab$diag_3) %in% c(800:999)] <- "Injury"
levels(diab$diag_3)[levels(diab$diag_3) %in% c(710:739)] <- "Musculoskeletal"
levels(diab$diag_3)[levels(diab$diag_3) %in% c(580:629,788)] <- "Genitourinary"
levels(diab$diag_3)[levels(diab$diag_3) %in% c(140:239,780,781,784,790:799,240:249,251:279,680:709,782,001:139)] <- "Neoplasms"
Defined=c("Circulatory","Respiratory","Digestive","Diabetes","Injury","Musculoskeletal","Genitourinary","Neoplasms")
levels(diab$diag_3)[!(levels(diab$diag_3) %in% Defined)] <- "Other"
table(diab$diag_3)#Grouped levels by ICD9 codes

table(diab$payer_code)
levels(diab$payer_code)[levels(diab$payer_code)=='?']='Unk'

table(diab$weight)

levels(diab$weight)[levels(diab$weight)=='?'] <- "Unk"

table(train$medical_specialty)

levels(diab$medical_specialty)[levels(diab$medical_specialty)=='?']='Unk'
levels(diab$medical_specialty)[levels(diab$medical_specialty)!= 'Nephrology' & levels(diab$medical_specialty)!= 'Unk' & levels(diab$medical_specialty)!= 'Orthopedics' & levels(diab$medical_specialty)!= 'Orthopedics-Reconstructive' & levels(diab$medical_specialty)!='Radiologist' & levels(diab$medical_specialty)!='Family/GeneralPractice' & levels(diab$medical_specialty)!='Surgery-General' & levels(diab$medical_specialty)!='Emergency/Trauma' & levels(diab$medical_specialty)!='Cardiology' & levels(diab$medical_specialty)!='InternalMedicine'] <- "Other"



#reducing levels



levels(diab$A1Cresult)

table(diab$A1Cresult)


#Removing Rows

diab = diab[diab$discharge_disposition_id != '11',]

ncol(diab)
nrow(diab)



#Building Training and Testing models
set.seed(123)

grp = runif(nrow(diab))
diab = diab[order(grp),]



ind = sample(seq_len(nrow(diab)), floor(nrow(diab)*0.7)   )
train = diab[ind,]
test = diab[-ind,]


train_x = train[,1:45]
train_y = train[,46]


head(train_x,3)
head(train_y,3)


str(diab)




ncol(diab)
rf1 = randomForest(train_x, factor(train_y) )
summary(rf1)

str(diab)







pdct_rf1 = predict(rf1, test)
pdct_rf1
table(predicted=pdct_rf1,actual=test$readmitted)

confusionMatrix(pdct_rf1,test$readmitted,positive = "positive")

ncol(diab)
colnames(diab)




#Feature selection
importance(rf1)
varImpPlot(rf1)





#model number 2

train_x$metformin.pioglitazone=NULL
train$metformin.pioglitazone=NULL
test$metformin.pioglitazone=NULL


train_x$metformin.rosiglitazone=NULL
train$metformin.rosiglitazone=NULL
test$metformin.rosiglitazone=NULL

ncol(train)
ncol(train_x)
ncol(test)


head(train_x,3)
head(train_y,3)

rf2 = randomForest(train_x, factor(train_y) )
summary(rf2)

pdct_rf2 = predict(rf2, test)
pdct_rf2
table(predicted=pdct_rf2,actual=test$readmitted)
confusionMatrix(pdct_rf2,test$readmitted,
                positive = "positive")

# it gives slightly better accuracy, better sesnitivity for specifically class <30


importance(rf2)
varImpPlot(rf1)
varUsed(rf1, by.tree = F, count=F)

#Feature selection
train_x$glimepiride.pioglitazone=NULL
train$glimepiride.pioglitazone=NULL
test$glimepiride.pioglitazone=NULL

train_x$acetohexamide=NULL
train$acetohexamide=NULL
test$acetohexamide=NULL

train_x$troglitazone=NULL
train$troglitazone=NULL
test$troglitazone=NULL

train_x$glipizide.metformin=NULL
train$glipizide.metformin=NULL
test$glipizide.metformin=NULL



ncol(train)







rf3 = randomForest(train_x, factor(train_y) )
summary(rf3)

pdct_rf3 = predict(rf3, test)
pdct_rf3
table(predicted=pdct_rf3,actual=test$readmitted)
confusionMatrix(pdct_rf3,test$readmitted,
                positive = "positive")




#model 3 the accuracy increases

importance(rf3)



train_x$trogl=NULL
train$troglitazone=NULL
test$troglitazone=NULL


rf1 = randomForest(train_x, factor(train_y) )
summary(rf1)

pdct_rf1 = predict(rf1, test)
pdct_rf1
table(predicted=pdct_rf1,actual=test$readmitted)
confusionMatrix(pdct_rf1,test$readmitted,
                positive = "positive")


importance(rf2)
varImpPlot(rf1)
varUsed(rf1, by.tree = F, count=F)

ncol(diab)

#model 4 accuracy decreases significantly by 10%
#NO
ind = sample(seq_len(nrow(diab)), floor(nrow(diab)*0.7)   )
train = diab[ind,]
test = diab[-ind,]

ncol(diab)

ncol(train_x)

rf3 = randomForest(train_x, factor(train_y) )
summary(rf3)

pdct_rf3 = predict(rf3, test)
pdct_rf3
table(predicted=pdct_rf3,actual=test$readmitted)
confusionMatrix(pdct_rf3,test$readmitted,
                positive = "positive")





















#--------------------------------------
#KNN
#error in model regarding NAs in coerrence
ncol(diab)
sample_size = floor(0.7*nrow(diab))
sample_ind = sample(seq_len(nrow(diab)), sample_size)
train = diab[sample_ind,]
test=diab[-sample_ind,]
ncol(train)
ncol(test)

traintarget=train$readmitted
testtarget=test$readmitted

train$readmitted=NULL
test$readmitted=NULL

str(diab)

model_knncv = knn.cv(train, traintarget, k=3)
cv_accuracy[i] = length(which(model_knncv==traintarget, T)) / length(model_knncv)
cv_accuracy


predict_target = knn(train, test, traintarget, k=3)





#decision tree
# poorest model with no True Positives

table(test$readmitted)
table(train$readmitted)
dt_readmitted = rpart(readmitted ~., method="class", data=train)

pdct = predict(dt_readmitted, test, type="class")


ncol(test)
confusionMatrix(test[,40],pdct)



#SVM model
# using radial kernel


model_lin = svm(readmitted~., data=train, kernel="radial")
summary(model_lin)

#unable to process the model using this algorithm