#if 0
	#define	PRINT_LOG(log_file, message) {	\
					print_log(log_file, \
							message,	\
							program_name,	\
							__FILE__, 	\
							__LINE__,	\
							__FUNCTION__);	\
		}



	#define	PRINT_ERR(log_file, message) {	\
				print_err(log_file, 	\
							message,	\
							program_name);	\
		}

#endif

#if 1
	#define	PRINT_LOG(log_file, message) { }



	#define	PRINT_ERR(log_file, message) { } 
#endif
