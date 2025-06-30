#Generate HR, RR z-scores
library(rriskDistributions)
library(tidyverse)


### Do HR, RR correction
#Tables from Fleming
age = c(0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 15, 25)
RR_cutoffs <- data.frame(age, 
                         first = c(25, 24, 23, 22, 21, 19, 18, 17, 17, 16, 14, 12, 11), 
                         tenth = c(34, 33, 31, 30, 28, 25, 22, 21, 20, 18, 16, 15, 13), 
                         quarter = c(40, 38, 36, 35, 32, 29, 25, 23, 21, 20, 18, 16, 15),
                         median = c(43, 41, 39, 37, 35, 31, 28, 25, 23, 21, 19, 18, 16), 
                         threequarter = c(52, 49, 47, 45, 42, 36, 31, 27, 25, 23, 21, 19, 18), 
                         ninetieth = c(57, 55, 52, 50, 46, 40, 34, 29, 27, 24, 22, 21, 19), 
                         ninety9th = c(66, 64, 61, 58, 53, 46, 38, 33, 29, 27, 25, 23, 22))

HR_cutoffs <- data.frame(age,
                         first = c(107, 104, 98, 93, 88, 82, 76, 70, 65, 59, 52, 47, 43), 
                         tenth = c(123, 120, 114, 109, 103, 98, 92, 86, 81, 74, 67, 62, 58), 
                         quarter = c(133, 129, 123, 118, 112, 106, 100, 94, 89, 82, 75, 69, 65), 
                         median = c(143, 140, 134, 128, 123, 116, 110, 104, 98, 91, 84, 78, 73), 
                         threequarter = c(154, 150, 143, 137, 132, 126, 119, 113, 108, 101, 93, 87, 83), 
                         ninetieth = c(164, 159, 152, 145, 140, 135, 128, 123, 117, 111, 103, 96, 92), 
                         ninety9th = c( 181, 175, 168, 161, 156, 149, 142, 136, 131, 123, 115, 108, 104))

###Start with MAP
#Now read in BP tables
map_centiles <- read.csv('~/Documents/Masters/Course materials/Project/PICU_project/MAP_centiles.csv')
#Tidy up ages
map_centiles$Age <- c(NA, rep(1:17, each = 3))
#Tidy up colnames
names(map_centiles) <- c('Age', 'Centile_for_height', 5,5,25,25,50,50,75,75,95,95)

#Split into male and female
map_centiles_male <- map_centiles[2:dim(map_centiles)[1], #lose the first row (which has gender)
                                  c(1,2,which(map_centiles[1,] == 'M'))] #Only male columns + first 2
map_centiles_female <- map_centiles[2:dim(map_centiles)[1], #lose the first row (which has gender)
                                  c(1,2,which(map_centiles[1,] == 'F'))] #Only male columns + first 2

map_centiles_male <- map_centiles_male[,c('Age', 'Centile_for_height', '50')] %>%
  pivot_wider(names_from = 'Centile_for_height', values_from = '50') %>%
  apply(2, as.numeric)

map_centiles_female <- map_centiles_female[,c('Age', 'Centile_for_height', '50')] %>%
  pivot_wider(names_from = 'Centile_for_height', values_from = '50')  %>%
  apply(2, as.numeric)


### Now repeat for SBP
sbp_centiles <- read.csv('~/Documents/Masters/Course materials/Project/PICU_project/SBP_centiles.csv')
sbp_centiles_male <- sbp_centiles[2:dim(sbp_centiles)[1], #lose the first row (which has gender)
                                  c(1,which(sbp_centiles[1,] == 'M'))]  %>%
                    apply(2, as.numeric)
sbp_centiles_male <- sbp_centiles_male[, c('Age', 'X50')]
sbp_centiles_female <- sbp_centiles[2:dim(sbp_centiles)[1], #lose the first row (which has gender)
                                  c(1,which(sbp_centiles[1,] == 'F'))]  %>%
                    apply(2, as.numeric)
sbp_centiles_female <- sbp_centiles_female[, c('Age', 'X.2')]

### Now read in BP centiles 
fourth_report_centiles_boys <- read.csv('~/Documents/Masters/Course materials/Project/PICU_project/BP_centiles_boys.csv')
fourth_report_centiles_girls <- read.csv('~/Documents/Masters/Course materials/Project/PICU_project/BP_centiles_girls.csv')

#Start with SBP
fourth_sbp_centiles_male <- fourth_report_centiles_boys[,c('Age', 'BP_centile', 'X50th')]
fourth_sbp_centiles_female <- fourth_report_centiles_girls[,c('Age', 'BP_centile', 'X50th')]
fourth_sbp_centiles_male$BP_centile <- as.numeric(gsub('([[:digit:]]{2})([[:lower:]]){2}', '\\1', fourth_sbp_centiles_male$BP_centile))
fourth_sbp_centiles_female$BP_centile <- as.numeric(gsub('([[:digit:]]{2})([[:lower:]]){2}', '\\1', fourth_sbp_centiles_female$BP_centile))
fourth_sbp_centiles_male <- fourth_sbp_centiles_male %>% 
  pivot_wider(names_from = 'BP_centile', values_from ='X50th')
fourth_sbp_centiles_female <- fourth_sbp_centiles_female %>% 
  pivot_wider(names_from = 'BP_centile', values_from ='X50th')

#Combine SBP centiles from fourth report and Zaritsky
sbp_centiles_male <- cbind(sbp_centiles_male,fourth_sbp_centiles_male[,2:5])
sbp_centiles_female <- cbind(sbp_centiles_female,fourth_sbp_centiles_female[,2:5])

#Now DBP from fourth report
fourth_dbp_centiles_male <- fourth_report_centiles_boys[,c('Age', 'BP_centile', 'X50th.1')]
fourth_dbp_centiles_female <- fourth_report_centiles_girls[,c('Age', 'BP_centile', 'X50th.1')]
fourth_dbp_centiles_male$BP_centile <- as.numeric(gsub('([[:digit:]]{2})([[:lower:]]){2}', '\\1', fourth_dbp_centiles_male$BP_centile))
fourth_dbp_centiles_female$BP_centile <- as.numeric(gsub('([[:digit:]]{2})([[:lower:]]){2}', '\\1', fourth_dbp_centiles_female$BP_centile))
fourth_dbp_centiles_male <- fourth_dbp_centiles_male %>% 
  pivot_wider(names_from = 'BP_centile', values_from ='X50th.1')
fourth_dbp_centiles_female <- fourth_dbp_centiles_female %>% 
  pivot_wider(names_from = 'BP_centile', values_from ='X50th.1')

#Now derive DBP 5th centile using zaritsky data
#Using the above 2 calculate diastolic blood pressure
#map  = diastolic bp + (1/3)*(sbp - diastolic bp)
#map = diastolic bp + 1/3sbp - 1/3diabp
#map - 1/3sbp = diastolic bp - 1/3diabp = 2/3 diabp
#3/2 (map - 1/3sbp) = diastolic bp
dbp_fifth_centile_male = (3/2)*(map_centiles_male[, 2] - (1/3)*sbp_centiles_male[, c(2)])
dbp_fifth_centile_female = (3/2)*(map_centiles_female[, 2] - (1/3)*sbp_centiles_female[, c(2)])
dbp_centiles_male = data.frame(Age = 1:17, 
                               '5' = dbp_fifth_centile_male)
dbp_centiles_female = data.frame(Age = 1:17, 
                                 '5' = dbp_fifth_centile_female)
dbp_centiles_male = cbind(dbp_centiles_male, fourth_dbp_centiles_male[,2:5])
dbp_centiles_female = cbind(dbp_centiles_female, fourth_dbp_centiles_female[,2:5])

ggplot() + geom_point(aes(x = 1:17, y = sbp_centiles_male[,2]), col = 'blue') + geom_line(aes(x = 1:17, y = sbp_centiles_male[,2]), col = 'blue') + 
  geom_point(aes(x = 1:17, y = sbp_centiles_male[,3]), col = 'red') + geom_line(aes(x = 1:17, y = sbp_centiles_male[,3]), col = 'red') + 
  geom_point(aes(x = 1:17, y = sbp_centiles_male[,4]), col = 'green') + geom_line(aes(x = 1:17, y = sbp_centiles_male[,4]), col = 'green') +
  xlab('Age') + ylab('Systolic BP') + theme_bw()


#Now make function to make means and SD
get_meansd <- function(q, p){
  #Function to return mean and sd
  a <- get.norm.par(p, q)
  c(a[1], a[2])
}

#Should probably export these curves and use them
RR_meansd <- t(apply(RR_cutoffs[, 2:dim(HR_cutoffs)[2]], 1, get_meansd, p = c(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)))
RR_meansd <- data.frame(age, mean = RR_meansd[,1], sd = RR_meansd[,2])
HR_meansd <- t(apply(HR_cutoffs[, 2:dim(HR_cutoffs)[2]], 1, get_meansd, p = c(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)))
HR_meansd <- data.frame(age, mean = HR_meansd[,1], sd = HR_meansd[,2])

#SBP
SBP_meansd_male <- t(apply(sbp_centiles_male[, 2:dim(sbp_centiles_male)[2]], 1, get_meansd, p = c(0.05, 0.5, 0.90, 0.95, 0.99)))
SBP_meansd_male <- data.frame(age = sbp_centiles_male[,1], mean = SBP_meansd_male[,1], sd = SBP_meansd_male[,2])
SBP_meansd_female <- t(apply(sbp_centiles_female[, 2:dim(sbp_centiles_female)[2]], 1, get_meansd, p = c(0.05, 0.5, 0.90, 0.95, 0.99)))
SBP_meansd_female <- data.frame(age = sbp_centiles_female[,1], mean = SBP_meansd_female[,1], sd = SBP_meansd_female[,2])

#DBP
DBP_meansd_male <- t(apply(dbp_centiles_male[, 2:dim(dbp_centiles_male)[2]], 1, get_meansd, p = c(0.05, 0.5, 0.90, 0.95, 0.99)))
DBP_meansd_male <- data.frame(age = dbp_centiles_male[,1], mean = DBP_meansd_male[,1], sd = DBP_meansd_male[,2])
DBP_meansd_female <- t(apply(dbp_centiles_female[, 2:dim(dbp_centiles_female)[2]], 1, get_meansd, p = c(0.05, 0.5, 0.90, 0.95, 0.99)))
DBP_meansd_female <- data.frame(age = dbp_centiles_female[,1], mean = DBP_meansd_female[,1], sd = DBP_meansd_female[,2])

#MAP
MAP_meansd_male <- t(apply(map_centiles_male[, 2:dim(map_centiles_male)[2]], 1, get_meansd, p = c(0.05, 0.5, 0.95)))
MAP_meansd_male <- data.frame(age = map_centiles_male[,1], mean = MAP_meansd_male[,1], sd = MAP_meansd_male[,2])
MAP_meansd_female <- t(apply(map_centiles_female[, 2:dim(map_centiles_female)[2]], 1, get_meansd, p = c(0.05, 0.5, 0.95)))
MAP_meansd_female <- data.frame(age = map_centiles_female[,1], mean = MAP_meansd_female[,1], sd = MAP_meansd_female[,2])

#Estimate values for <1 using straight line fitted for 4 and under
#For MAP males
MAP_lm_male <- lm(mean ~age, data = MAP_meansd_male[1:4,])
predicted_MAP_male <- data.frame(age = 0,
                                 mean = predict(MAP_lm_male, data.frame(age = 0)),
                                 sd = MAP_meansd_male[1, 'sd'])
MAP_meansd_male <- rbind(predicted_MAP_male, MAP_meansd_male)

#Females
MAP_lm_female <- lm(mean ~age, data = MAP_meansd_female[1:4,])
predicted_MAP_female <- data.frame(age = 0,
                                 mean = predict(MAP_lm_female, data.frame(age = 0)),
                                 sd = MAP_meansd_female[1, 'sd'])
MAP_meansd_female <- rbind(predicted_MAP_female, MAP_meansd_female)

#DBP
DBP_lm_male <- lm(mean ~age, data = DBP_meansd_male[1:4,])
predicted_DBP_male <- data.frame(age = 0,
                                 mean = predict(DBP_lm_male, data.frame(age = 0)),
                                 sd = DBP_meansd_male[1, 'sd'])
DBP_meansd_male <- rbind(predicted_DBP_male, DBP_meansd_male)

#Females
DBP_lm_female <- lm(mean ~age, data = DBP_meansd_female[1:4,])
predicted_DBP_female <- data.frame(age = 0,
                                   mean = predict(DBP_lm_female, data.frame(age = 0)),
                                   sd = DBP_meansd_female[1, 'sd'])
DBP_meansd_female <- rbind(predicted_DBP_female, DBP_meansd_female)

#SBP
SBP_lm_male <- lm(mean ~age, data = SBP_meansd_male[1:4,])
predicted_SBP_male <- data.frame(age = 0,
                                 mean = predict(SBP_lm_male, data.frame(age = 0)),
                                 sd = SBP_meansd_male[1, 'sd'])
SBP_meansd_male <- rbind(predicted_SBP_male, SBP_meansd_male)

#Females
SBP_lm_female <- lm(mean ~age, data = SBP_meansd_female[1:4,])
predicted_SBP_female <- data.frame(age = 0,
                                   mean = predict(SBP_lm_female, data.frame(age = 0)),
                                   sd = SBP_meansd_female[1, 'sd'])
SBP_meansd_female <- rbind(predicted_SBP_female, SBP_meansd_female)

write.csv(SBP_meansd_female, '~/Documents/Masters/Course materials/Project/PICU_project/files/SBP_meansd_female.csv')
write.csv(SBP_meansd_male, '~/Documents/Masters/Course materials/Project/PICU_project/files/SBP_meansd_male.csv')
write.csv(DBP_meansd_female, '~/Documents/Masters/Course materials/Project/PICU_project/files/DBP_meansd_female.csv')
write.csv(DBP_meansd_male, '~/Documents/Masters/Course materials/Project/PICU_project/files/DBP_meansd_male.csv')
write.csv(MAP_meansd_female, '~/Documents/Masters/Course materials/Project/PICU_project/files/MAP_meansd_female.csv')
write.csv(MAP_meansd_male, '~/Documents/Masters/Course materials/Project/PICU_project/files/MAP_meansd_male.csv')
write.csv(RR_meansd, '~/Documents/Masters/Course materials/Project/PICU_project/files/RR_meansd.csv')
write.csv(HR_meansd, '~/Documents/Masters/Course materials/Project/PICU_project/files/HR_meansd.csv')
