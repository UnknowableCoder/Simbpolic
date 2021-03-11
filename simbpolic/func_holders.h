#ifndef SIMBPOLIC_FUNC_HOLDERS
#define SIMBPOLIC_FUNC_HOLDERS

namespace Simbpolic
{
  namespace internals
    {
    template <class ... members> class holder_impl;

    //tag and may_derive are to prevent -Winacessible-base warnings...
    //Since we're doing template-based shenanigans anyway, it doesn't matter.
    template <class member, indexer tag, bool really_hold, bool may_derive> class holder_helper;

    
    template <class member, indexer tag> class holder_helper<member, tag, true, true>:
    public SymHoldsValues
    {
      private:
      member x;
      
      public:
            
      SIMBPOLIC_CUDA_HOS_DEV constexpr holder_helper() = default;
      
      SIMBPOLIC_CUDA_HOS_DEV constexpr holder_helper(const member& val): x(val)
      {
      }
      
      SIMBPOLIC_CUDA_HOS_DEV inline constexpr const member& get() const
      {
          return x;
      }
    };

    template <class member, indexer tag> class holder_helper<member, tag, true, false>
    {
      private:
      member x;
      
      public:
      
      SIMBPOLIC_CUDA_HOS_DEV constexpr holder_helper() = default;
      
      SIMBPOLIC_CUDA_HOS_DEV constexpr holder_helper(const member& val): x(val)
      {
      }
      
      SIMBPOLIC_CUDA_HOS_DEV inline constexpr const member& get() const
      {
          return x;
      }
    };
    
    template <class member, indexer tag, bool any> class holder_helper<member, tag, false, any>
    {
      public:
            
      SIMBPOLIC_CUDA_HOS_DEV constexpr holder_helper() = default;
      
      SIMBPOLIC_CUDA_HOS_DEV constexpr holder_helper(const member& val)
      {
      }
      
      SIMBPOLIC_CUDA_HOS_DEV inline constexpr member get() const
      {
          return member{};
      }
    };

    template <class member> class holder_impl<member> : public holder_helper<member, 1, holds_values<member>, true>
    {
      public:
      
      SIMBPOLIC_CUDA_HOS_DEV constexpr holder_impl() = default;
      
      //using holder_helper<member, 1, holds_values<member>, true>::holder_helper;
      
      SIMBPOLIC_CUDA_HOS_DEV constexpr holder_impl(const member& m):
      holder_helper<member, 1, holds_values<member>, true>(m)
      {
      }
      
      template <indexer i>
      SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto get() const
      {
          return holder_helper<member, 1, holds_values<member>, true>::get();
      }
    };


    template <class member, class ... members> class holder_impl<member, members...> : 
    public holder_impl<members...>,
    public holder_helper<member, sizeof...(members) + 1, holds_values<member>, !holds_values< holder_impl<members...> > >
    {
      public:
      
      SIMBPOLIC_CUDA_HOS_DEV constexpr holder_impl() = default;
      
      SIMBPOLIC_CUDA_HOS_DEV constexpr holder_impl(const member& m, const members& ... mm):
      holder_helper<member, sizeof...(members) + 1, holds_values<member>, !holds_values< holder_impl<members...> > >(m),
      holder_impl<members...>(mm...)
      {
      }
      
      template <indexer i>
      SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto get() const
      {
        if constexpr (i == sizeof...(members) + 1)
          {
            return holder_helper<member, sizeof...(members) + 1, holds_values<member>, !holds_values< holder_impl<members...> >>::get();
          }
        else
          {
            return holder_impl<members...>::template get<i>();
          }
      }
    };
        
  }
  
  template <class ... funcs> class func_holder: internals::holder_impl<funcs...>
  {
    public:
    
    using internals::holder_impl<funcs...>::holder_impl;
      
    template <indexer i>
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto get() const
    {
      return internals::holder_impl<funcs...>::template get<sizeof...(funcs) - i>();
    }
    
  };
}

#endif
