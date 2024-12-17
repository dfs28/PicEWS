#!/usr/bin/env Rscript
#### Adjust BP, other parameters using childsds package

#Setup
library(childsds)

#Define a function for printing the date, time and what we are doing:
print_now <- function(statement){
  #Function that takes a statement and prints the date and time
  print(paste(statement, Sys.time(), sep = ': '))
}

#Read in the sheet
print_now('Reading in flowsheet')
flowsheet = read.csv('/store/DAMTP/dfs28/PICU_data/flowsheet_final_output_pSOFA.csv.gz', header = TRUE, sep = ',')
print_now('Flowsheet read')

#Will need to go through and only do corrections for that age range, then do the corretions for younger children


#Make corrections for weight
interpolated_weights = as.vector(flowsheet$interpolated_wt_kg)
Ages_yrs = as.vector(flowsheet$Age_yrs)
Sexes = as.vector(flowsheet$sex)
flowsheet$Weight_z_scores =  sds(interpolated_weights , 
            age = Ages_yrs, 
            sex = Sexes,
            male = 'M', 
            female = 'F', 
            ref = uk1990.ref, 
            item = 'weight', 
            type = 'SDS')
print(length(interpolated_weights))
print(length(Ages_yrs))
print(length(Sexes))

#Do those which are intersex as average of male and female
intersexes <- which(!Sexes %in% c('M', 'F'))
flowsheet$Weight_z_scores[intersexes] =  (sds(interpolated_weights[intersexes], 
            age = Ages_yrs[intersexes], 
            sex = rep('M', times = length(intersexes)),
            male = 'M', 
            female = 'F', 
            ref = uk1990.ref, 
            item = 'weight', 
            type = 'SDS') +
            sds(interpolated_weights[intersexes], 
            age = Ages_yrs[intersexes], 
            sex = rep('F', times = length(intersexes)),
            male = 'M', 
            female = 'F', 
            ref = kro.ref, 
            item = 'weight', 
            type = 'SDS'))/2


print_now('Weight corrected')

interpolated_heights = as.vector(flowsheet$interpolated_ht_m)
flowsheet$Height_z_scores =  sds(interpolated_heights, 
            age = Ages_yrs, 
            sex = Sexes,
            male = 'M', 
            female = 'F', 
            ref = uk1990.ref, 
            item = 'height', 
            type = 'SDS')

#Again average as above for sex = I
flowsheet$Height_z_scores[intersexes] =  (sds(interpolated_heights[intersexes], 
            age = Ages_yrs[intersexes], 
            sex = rep('M', times = length(intersexes)),
            male = 'M', 
            female = 'F', 
            ref = uk1990.ref, 
            item = 'height', 
            type = 'SDS') +
            sds(interpolated_heights[intersexes], 
            age = Ages_yrs[intersexes], 
            sex = rep('F', times = length(intersexes)),
            male = 'M', 
            female = 'F', 
            ref = uk1990.ref, 
            item = 'height', 
            type = 'SDS'))/2

#Where height not available use the weight z-scores for height
no_height <- which(is.na(flowsheet$Height_z_scores))
flowsheet$Height_z_scores[no_height] <- flowsheet$Weight_z_scores[no_height]

#For these patients need to return an imputed height where missing:
#Make table of percentages
perc_tab <- childsds::make_percentile_tab(
  ref = uk1990.ref, item = "height", age=0:25,
  perc=c(5,50,95), include.pars= FALSE)

#Function to convert z-score back to height
get_height <- function(z_score, age, sex, lookup_table){
  
  #Filter lookup table by sex
  if (sex == 'F') {
    lookup_table <- lookup_table[lookup_table$sex == 'female', ]
  } else if (sex == 'M') {
    lookup_table <- lookup_table[lookup_table$sex == 'male', ]
  } else {
    lookup_table <- (lookup_table[lookup_table$sex == 'female', 2:5] + lookup_table[lookup_table$sex == 'male', 2:5])/2
  }

  #Get age row
  age_row = sort(which(lookup_table$age < age), decr = T)[1]

  #Now use qnorm and pnorm to return the height based on that z-score
  qnorm(pnorm(z_score), lookup_table$perc_50_0[age_row], 
    (lookup_table$perc_50_0[age_row] - lookup_table$perc_05_0[age_row])/1.65)
}

#Now run on patients where there was originally no height (have to use a wrapper to make this work)
flowsheet$interpolated_ht_m[no_height] <- sapply(no_height, 
                                                g <- function(x) 
                                                  {get_height(flowsheet$Height_z_scores[x], 
                                                    Ages_yrs[x], 
                                                    Sexes[x], 
                                                    perc_tab)/100}) #divide by 100 to get height in m

#For children of age 0 (where the above function is returning NA):
#Now run on patients where there was originally no height (have to use a wrapper to make this work)
no_height_interp <- which(is.na(flowsheet$interpolated_ht_m))
flowsheet$interpolated_ht_m[no_height_interp] <- sapply(no_height_interp, 
                                                g <- function(x) 
                                                  {get_height(flowsheet$Height_z_scores[x], 
                                                    Ages_yrs[x] + 0.0001, 
                                                    Sexes[x], 
                                                    perc_tab)/100}) #divide by 100 to get height in m


#Need to read in the other things
HR_meansd = read.csv('/mhome/damtp/q/dfs28/Project/PICU_project/files/HR_meansd.csv', sep = ',', header = TRUE)
RR_meansd = read.csv('/mhome/damtp/q/dfs28/Project/PICU_project/files/RR_meansd.csv', sep = ',', header = TRUE)
MAP_male_meansd = read.csv('/mhome/damtp/q/dfs28/Project/PICU_project/files/MAP_meansd_male.csv', sep = ',', header = TRUE)
MAP_female_meansd = read.csv('/mhome/damtp/q/dfs28/Project/PICU_project/files/MAP_meansd_female.csv', sep = ',', header = TRUE)
DBP_male_meansd = read.csv('/mhome/damtp/q/dfs28/Project/PICU_project/files/DBP_meansd_male.csv', sep = ',', header = TRUE)
DBP_female_meansd = read.csv('/mhome/damtp/q/dfs28/Project/PICU_project/files/DBP_meansd_female.csv', sep = ',', header = TRUE)
SBP_male_meansd = read.csv('/mhome/damtp/q/dfs28/Project/PICU_project/files/SBP_meansd_male.csv', sep = ',', header = TRUE)
SBP_female_meansd = read.csv('/mhome/damtp/q/dfs28/Project/PICU_project/files/SBP_meansd_female.csv', sep = ',', header = TRUE)

#Now calculate z-scores
calc_zscore <- function(row, sheet, input_col, age_col, scortab, sex = NA, scortab_f = NA) {
  #Function to calculate z-scores from table of mean and sd
  
  #Get age range
  if (is.na(sex)) {  
    age_range = which(scortab$age > sheet[row, age_col])[1]
    if (is.na(age_range)) {age_range = dim(scortab)[1]}
  
    #Get absolute distance from mean
    dev = sheet[row, input_col] - scortab$mean[age_range]
  
    #Get distance from med
    return(dev/scortab$sd[age_range])
  } else {
    #Need to invert the sort as its now older than not less than
    age_range = sort(which(scortab$age < sheet[row, age_col]), decreasing = T)[1]

    #For children whose age is 0.0:
    if (is.na(age_range)) {
      age_range = 1
    }

    #Do boys
    if (sheet[row, sex] == 'M') {
      #Get absolute distance from mean
      dev = sheet[row, input_col] - scortab$mean[age_range]
  
      #Get distance from med
      return(dev/scortab$sd[age_range])
    } else if (sheet[row, sex] == 'M') { 
      #Do girls  
      #Get absolute distance from mean
      dev = sheet[row, input_col] - scortab_f$mean[age_range]
  
      #Get distance from med
      return(dev/scortab_f$sd[age_range])
    } else {
      #Do I
      #Get absolute distance from mean
      scortab_f = (scortab_f + scortab)/2
      dev = sheet[row, input_col] - scortab_f$mean[age_range]
  
      #Get distance from med
      return(dev/scortab_f$sd[age_range])
    }
  }
}  


calc_zscore(1, flowsheet, 'HR', 'Age_yrs', HR_meansd)
flowsheet$HR_zscore = sapply(1:dim(flowsheet)[1], calc_zscore, sheet = flowsheet, input_col = 'HR', age_col = 'Age_yrs', scortab = HR_meansd)
print_now('HR corrected')
flowsheet$RR_zscore = sapply(1:dim(flowsheet)[1], calc_zscore, sheet = flowsheet, input_col = 'RR', age_col = 'Age_yrs', scortab = RR_meansd)
print_now('RR corrected')
flowsheet$DBP_zscore = sapply(1:dim(flowsheet)[1], calc_zscore, sheet = flowsheet, input_col = 'DiaBP',
  age_col = 'Age_yrs', scortab = DBP_male_meansd, scortab_f = DBP_female_meansd, sex = 'sex')
print_now('DBP corrected')
flowsheet$SBP_zscore = sapply(1:dim(flowsheet)[1], calc_zscore, sheet = flowsheet, input_col = 'SysBP',
  age_col = 'Age_yrs', scortab = SBP_male_meansd, scortab_f = SBP_female_meansd, sex = 'sex')
print_now('SBP corrected')
flowsheet$MAP_zscore = sapply(1:dim(flowsheet)[1], calc_zscore, sheet = flowsheet, input_col = 'MAP',
  age_col = 'Age_yrs', scortab = MAP_male_meansd, scortab_f = MAP_female_meansd, sex = 'sex')

print_now('MAP corrected')

write.csv(flowsheet, '/store/DAMTP/dfs28/PICU_data/flowsheet_zscores.csv')